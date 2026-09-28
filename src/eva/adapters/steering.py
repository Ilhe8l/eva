"""Messages the user sends while Eva is working, delivered at her next step."""

import threading
from collections import defaultdict

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.config import get_config

STEERING_PREFIX = "[sent while you were working]"


class SteeringInbox:
    """Per-thread mailbox, written by the terminal and read by the agent graph."""

    def __init__(self) -> None:
        self._messages: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def post(self, thread_id: str, text: str) -> None:
        with self._lock:
            self._messages[thread_id].append(text)

    def take(self, thread_id: str) -> list[str]:
        with self._lock:
            return self._messages.pop(thread_id, [])


class SteeringMiddleware(AgentMiddleware):
    """Before each model call, adds any messages the user sent to this thread meanwhile."""

    def __init__(self, inbox: SteeringInbox) -> None:
        super().__init__()
        self.inbox = inbox

    def before_model(self, state, runtime) -> dict | None:
        thread_id = get_config().get("configurable", {}).get("thread_id")
        messages = self.inbox.take(thread_id) if thread_id else []
        if not messages:
            return None
        return {"messages": [HumanMessage(content=f"{STEERING_PREFIX} {text}") for text in messages]}
