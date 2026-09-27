"""Eva-specific tools added to the Deep Agents built-ins."""

from langchain_core.tools import BaseTool, StructuredTool, tool

from eva.adapters.extensions import Extension, ExtensionStore
from eva.adapters.patches import PatchStore
from eva.adapters.voice import SpeechOutbox


def build_tools(speech: SpeechOutbox, extensions: ExtensionStore, patches: PatchStore) -> list[BaseTool]:
    @tool
    def speak_to_user(text: str) -> str:
        """Say one or two short sentences aloud to the user."""
        return speech.enqueue(text)

    @tool
    def propose_tool(name: str, description: str, source: str) -> str:
        """Stage a Python tool for human review. It reads JSON {"input": str} on stdin."""
        try:
            return extensions.propose(name, description, source)
        except (ValueError, SyntaxError) as exc:
            return f"Proposal rejected: {exc}"

    @tool
    def list_eva_tools() -> str:
        """List active and proposed extension tools."""
        active = ", ".join(item.name for item in extensions.list_active()) or "none"
        proposed = ", ".join(extensions.list_proposals()) or "none"
        return f"Active: {active}\nProposed: {proposed}"

    @tool
    def propose_source_patch(name: str, description: str, diff: str) -> str:
        """Stage a standard Git diff of src/eva or tests for human review."""
        try:
            return patches.propose(name, description, diff)
        except ValueError as exc:
            return f"Patch proposal rejected: {exc}"

    @tool
    def list_eva_patches() -> str:
        """List staged source patches waiting for review."""
        return ", ".join(patches.list()) or "No patches staged"

    return [
        speak_to_user,
        propose_tool,
        list_eva_tools,
        propose_source_patch,
        list_eva_patches,
        *(_extension_tool(extensions, item) for item in extensions.list_active()),
    ]


def _extension_tool(extensions: ExtensionStore, extension: Extension) -> BaseTool:
    def run(input: str) -> str:
        return extensions.run(extension, input)

    return StructuredTool.from_function(func=run, name=extension.name, description=extension.description)
