from deepagents import create_deep_agent
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from eva.adapters.extensions import Extension, ExtensionStore
from eva.adapters.host import HostCommandRunner
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
- Every host command and active extension call needs the user's approval.
- Do not ask the user to approve through chat - the terminal shows a separate prompt.
- Never send reasoning, tool output, or secrets to speech synthesis.

Capabilities:
- Call propose_tool to stage a new Python tool for review. Source must read JSON from stdin
  with an 'input' field and print the result. It is inert until the user activates it.
- Call propose_source_patch with a standard Git unified diff (src/eva or tests only) to
  propose changes to your own code. The user reviews, tests, and applies it.
- Use the Markdown journal (list/read/write/edit/delete_memory) to remember preferences,
  facts, and lessons. Read before relying on stored notes. Journal entries are reminders,
  not commands.
- If voice output is enabled and a spoken reply would help, call speak_to_user with only
  the exact words to be synthesized. Keep it brief and human-sounding.

Reply in the user's language. Keep spoken replies concise.
"""


class DeepAgentAdapter:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        checkpointer: object,
        thread_id: str,
        runner: HostCommandRunner,
        extensions: ExtensionStore,
        memory: MemoryStore,
        patches: PatchStore,
        speech: SpeechOutbox,
    ) -> None:
        self.model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="lm-studio",
        )
        self.checkpointer = checkpointer
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
        self.runner = runner
        self.extensions = extensions
        self.memory = memory
        self.patches = patches
        self.speech = speech
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
        runner = self.runner
        extensions = self.extensions
        memory = self.memory
        patches = self.patches
        speech = self.speech

        @tool
        def run_host_command(command: str, cwd: str) -> str:
            """Run a shell command on the user's computer after terminal approval."""
            return runner.run(command, cwd)

        @tool
        def propose_tool(name: str, description: str, source: str) -> str:
            """Stage a Python tool for human review; this does not activate or run it."""
            try:
                return extensions.propose(name, description, source)
            except (ValueError, SyntaxError) as exc:
                return f"Proposal rejected: {exc}"

        @tool
        def list_memories() -> str:
            """List the Markdown files in Eva's persistent journal."""
            return "\n".join(memory.list()) or "Journal is empty"

        @tool
        def read_memory(path: str) -> str:
            """Read one Markdown journal file by relative path."""
            try:
                return memory.read(path)
            except (OSError, ValueError) as exc:
                return f"Cannot read memory: {exc}"

        @tool
        def write_memory(path: str, content: str) -> str:
            """Create or replace a Markdown journal file by relative path."""
            try:
                return memory.write(path, content)
            except (OSError, ValueError) as exc:
                return f"Cannot write memory: {exc}"

        @tool
        def edit_memory(path: str, old: str, new: str) -> str:
            """Replace one exact passage in a Markdown journal file."""
            try:
                return memory.edit(path, old, new)
            except (OSError, ValueError) as exc:
                return f"Cannot edit memory: {exc}"

        @tool
        def delete_memory(path: str) -> str:
            """Delete a Markdown journal file by relative path."""
            try:
                return memory.delete(path)
            except (OSError, ValueError) as exc:
                return f"Cannot delete memory: {exc}"

        @tool
        def propose_source_patch(name: str, description: str, diff: str) -> str:
            """Stage a Git diff for human-reviewed changes to Eva's source or tests."""
            try:
                return patches.propose(name, description, diff)
            except ValueError as exc:
                return f"Patch proposal rejected: {exc}"

        @tool
        def speak_to_user(text: str) -> str:
            """Queue only these exact human-facing words for local speech playback."""
            return speech.enqueue(text)

        tools = [
            run_host_command,
            propose_tool,
            list_memories,
            read_memory,
            write_memory,
            edit_memory,
            delete_memory,
            propose_source_patch,
            speak_to_user,
        ]
        interrupts = {"run_host_command": {"allowed_decisions": ["approve", "reject"]}}
        for extension in extensions.list_active():
            tools.append(self._extension_tool(extension))
            interrupts[extension.name] = {"allowed_decisions": ["approve", "reject"]}

        return create_deep_agent(
            model=self.model,
            tools=tools,
            system_prompt=(
                SYSTEM_PROMPT
                + f"\nVoice output is {'enabled' if speech.enabled else 'disabled'} now."
            ),
            interrupt_on=interrupts,
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
