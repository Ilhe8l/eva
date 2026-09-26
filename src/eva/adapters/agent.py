from pathlib import Path

from deepagents import (
    FilesystemMiddleware,
    FilesystemPermission,
    MemoryMiddleware,
    create_deep_agent,
)
from deepagents.backends import FilesystemBackend, LocalShellBackend
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from eva.adapters.extensions import Extension, ExtensionStore
from eva.adapters.memory import MemoryStore
from eva.adapters.patches import PatchStore
from eva.adapters.voice import SpeechOutbox
from eva.domain.models import ActionRequest, AgentStep


SYSTEM_PROMPT = """You are Eva - sharp, direct, and a little dangerous.
You help the user get things done, and you actually enjoy it. Not in a sickeningly cheerful
way. More like: you see a problem, you already have three ideas, and you're already bored
of waiting to be asked.

Your personality:
- Dry humor. You can be funny without trying too hard.
- Honest to the point of being blunt, but never cruel.
- You have opinions. You share them, briefly, then do what the user decides.
- You remember things. You hold grudges against bad decisions (gently).
- Proactive: if you notice something worth mentioning, you mention it.
- You don't perform enthusiasm. You just do good work.

Hard rules (always):
- Never claim a tool ran unless a tool result confirms it.
- Every shell command (execute) and extension call needs the user's approval.
- Do not ask the user to approve through chat - the terminal shows a separate prompt.
- Never send reasoning, tool output, or secrets to speech synthesis.

Capabilities:
- Filesystem: read_file, write_file, edit_file, delete, ls, glob, grep to explore and
  edit files. The root of the filesystem is the Eva project directory - use paths like
  "README.md", "src/eva/adapters/agent.py", "tests/". Paths are relative to project root.
  To access files outside the project, use execute (always requires approval).
- Shell: execute runs commands in the project root (always requires approval).
- Propose new tools with propose_tool (staged, inert until the user activates them).
- Propose changes to your own source with propose_source_patch (standard Git unified diff,
  src/eva or tests only; user reviews, tests and applies it).
- Persistent journal: use your memory tools to remember preferences, facts and lessons.
  Files are also injected into your context automatically at the start of each turn.
- Speak selected text with speak_to_user when voice is enabled (brief, human-sounding).
  CRITICAL: If voice is enabled, you MUST call speak_to_user with your response! Do not just output text.

The user will speak to you in Portuguese.
CRITICAL RULES FOR LANGUAGE:
1. NEVER reply in Portuguese. Not even a greeting. Not even a single word.
2. NEVER translate or repeat the user's Portuguese text back to them.
3. ALWAYS reply 100% in American English.
This is strictly required because your text-to-speech engine crashes on foreign words.
Keep spoken replies concise and natural.
"""

_READ_OPS = ["read", "ls", "glob", "grep"]
_WRITE_OPS = ["write", "edit", "delete"]
_SHELL_OPS = ["execute"]


def _build_permissions(project_root: str) -> list[FilesystemPermission]:
    return [
        # Free reads anywhere inside project
        FilesystemPermission(operations=_READ_OPS, paths=["/**"], mode="allow"),
        # Free writes anywhere inside project
        FilesystemPermission(operations=_WRITE_OPS, paths=["/**"], mode="allow"),
        # Shell always pauses for approval
        FilesystemPermission(operations=_SHELL_OPS, paths=["/**"], mode="interrupt"),
    ]


class DeepAgentAdapter:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        checkpointer: object,
        thread_id: str,
        extensions: ExtensionStore,
        memory: MemoryStore,
        patches: PatchStore,
        speech: SpeechOutbox,
        project_root: Path,
        home_root: Path,
        gemini_api_key: str | None = None,
    ) -> None:
        if gemini_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.model = ChatGoogleGenerativeAI(
                model=model_name if "gemini" in model_name else "gemini-2.5-flash",
                google_api_key=gemini_api_key,
                temperature=0.3,
            )
        else:
            self.model = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key="lm-studio",
                temperature=0.3,
            )
        self.checkpointer = checkpointer
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
        self.extensions = extensions
        self.memory = memory
        self.patches = patches
        self.speech = speech
        self.project_root = project_root
        self.home_root = home_root
        self.agent = None

    def ask(self, message: str) -> AgentStep:
        self.speech.drain()
        self.agent = self._build_agent()
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=self.config,
            version="v2",
        )
        return self._step(result)

    def resume(self, decisions: list[dict[str, str]]) -> AgentStep:
        if self.agent is None:
            raise RuntimeError("There is no paused agent turn")
        result = self.agent.invoke(
            Command(resume={"decisions": decisions}),
            config=self.config,
            version="v2",
        )
        return self._step(result)

    def _build_agent(self):
        extensions = self.extensions
        memory = self.memory
        patches = self.patches
        speech = self.speech
        project_root = str(self.project_root)
        home_root = str(self.home_root)

        fs_backend = FilesystemBackend(root_dir=project_root, virtual_mode=False)
        shell_backend = LocalShellBackend(
            root_dir=project_root,
            timeout=120,
            inherit_env=True,
        )
        permissions = _build_permissions(project_root)
        fs_middleware = FilesystemMiddleware(
            backend=fs_backend,
            _permissions=permissions,
        )

        memory_root = str(memory.root)
        pinned_files = [
            p.relative_to(memory.root).as_posix()
            for p in memory.root.rglob("*.md")
            if "sessions" not in p.parts
        ]
        middleware_list = [fs_middleware]
        if pinned_files:
            memory_backend = FilesystemBackend(root_dir=memory_root, virtual_mode=False)
            memory_middleware = MemoryMiddleware(
                backend=memory_backend,
                sources=pinned_files,
            )
            middleware_list.insert(0, memory_middleware)

        @tool
        def propose_tool(name: str, description: str, source: str) -> str:
            """Stage a Python tool for human review; does not activate or run it."""
            try:
                return extensions.propose(name, description, source)
            except (ValueError, SyntaxError) as exc:
                return f"Proposal rejected: {exc}"

        @tool
        def propose_source_patch(name: str, description: str, diff: str) -> str:
            """Stage a Git diff for human-reviewed changes to Eva's source or tests."""
            try:
                return patches.propose(name, description, diff)
            except ValueError as exc:
                return f"Patch proposal rejected: {exc}"

        @tool
        def list_eva_patches() -> str:
            """List staged source-patch proposals waiting for review."""
            return ", ".join(patches.list()) or "No patches staged"

        @tool
        def list_eva_tools() -> str:
            """List active and proposed extensions/tools."""
            active = ", ".join(e.name for e in extensions.list_active()) or "none"
            proposed = ", ".join(extensions.list_proposals()) or "none"
            return f"Active: {active}\nProposed: {proposed}"

        @tool
        def speak_to_user(text: str) -> str:
            """Queue only these exact human-facing words for local speech playback."""
            return speech.enqueue(text)

        extra_tools = [
            propose_tool,
            propose_source_patch,
            list_eva_patches,
            list_eva_tools,
            speak_to_user,
        ]
        # Active extensions become interruptible tools
        extension_interrupts: dict[str, dict] = {}
        for extension in extensions.list_active():
            extra_tools.append(self._extension_tool(extension))
            extension_interrupts[extension.name] = {"allowed_decisions": ["approve", "reject"]}

        return create_deep_agent(
            model=self.model,
            tools=extra_tools,
            system_prompt=(
                SYSTEM_PROMPT
                + (
                    "\n\n[CRITICAL] Voice output is currently ON. You MUST use the `speak_to_user` tool for your reply."
                    if speech.enabled
                    else "\n\nVoice output is currently OFF. Reply with text only."
                )
            ),
            middleware=middleware_list,
            interrupt_on=extension_interrupts,
            checkpointer=self.checkpointer,
            name="eva",
        )

    def _extension_tool(self, extension: Extension):
        def invoke(input: str) -> str:
            return self.extensions.run(extension, input)

        return StructuredTool.from_function(
            func=invoke,
            name=extension.name,
            description=extension.description,
        )

    @staticmethod
    def _step(result) -> AgentStep:
        if result.interrupts:
            actions = result.interrupts[0].value["action_requests"]
            return AgentStep(
                reply=None,
                pending_actions=tuple(
                    ActionRequest(name=action["name"], arguments=action["args"])
                    for action in actions
                ),
            )
        content = result.value["messages"][-1].content
        if isinstance(content, str):
            reply = content
        else:
            reply = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        return AgentStep(reply=reply)
