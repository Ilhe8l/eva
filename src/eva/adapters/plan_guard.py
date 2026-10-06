"""Catches Eva stopping halfway through her own plan.

Small models sometimes describe the next step ("I'll now open the file") and
end the turn instead of calling the tool. The plan kept by `write_todos` makes
that visible: a reply without tool calls while steps are still pending or in
progress. The model call is then repeated once with a reminder, which is never
saved to the thread.
"""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config

UNFINISHED = {"pending", "in_progress"}
PLAN_NUDGE = (
    "[reminder] Your plan still has unfinished steps: {steps}. If you meant to keep working, call "
    "the tool for the next step now instead of describing it. If you are done, or need the user "
    "to decide something, update the plan with write_todos and then answer."
)


def unfinished_steps(state) -> list[str]:
    todos = state.get("todos") if isinstance(state, dict) else None
    return [str(todo.get("content", "")) for todo in todos or [] if todo.get("status") in UNFINISHED]


def stopped_early(response) -> bool:
    """Whether the model ended its turn with words only."""
    messages = response.result if hasattr(response, "result") else [response]
    last = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
    return last is not None and not last.tool_calls


class PlanFollowThroughMiddleware(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self._nudged: set[tuple[str, int]] = set()  # (thread, turn): one reminder per turn

    def wrap_model_call(self, request: ModelRequest, handler):
        response = handler(request)
        retry = self._retry(request, response)
        return handler(retry) if retry is not None else response

    async def awrap_model_call(self, request: ModelRequest, handler):
        response = await handler(request)
        retry = self._retry(request, response)
        return await handler(retry) if retry is not None else response

    def _retry(self, request: ModelRequest, response) -> ModelRequest | None:
        steps = unfinished_steps(request.state)
        if not steps or not stopped_early(response):
            return None
        turn = (_thread_id(), sum(isinstance(message, HumanMessage) for message in request.messages))
        if turn in self._nudged:
            return None
        self._nudged.add(turn)
        nudge = HumanMessage(content=PLAN_NUDGE.format(steps="; ".join(steps)))
        return request.override(messages=[*request.messages, nudge])


def _thread_id() -> str:
    try:
        return str(get_config().get("configurable", {}).get("thread_id", ""))
    except RuntimeError:  # outside a graph run
        return ""
