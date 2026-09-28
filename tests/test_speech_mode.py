from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eva.adapters.speech_mode import KICKOFF_NUDGE, PROGRESS_NUDGE, SpeechModeMiddleware
from eva.adapters.voice import SpeechChannel


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def step(name="execute", index=0):
    call_id = f"{name}-{index}"
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": call_id}]),
        ToolMessage(content="ok", tool_call_id=call_id),
    ]


def work(*steps):
    return [HumanMessage("do the thing"), *(message for s in steps for message in s)]


def middleware():
    clock = Clock()
    return SpeechModeMiddleware(SpeechChannel(), clock=clock), clock


def test_a_new_turn_gets_no_nudge_and_its_first_tool_step_gets_a_kickoff():
    guide, _ = middleware()
    assert guide.nudge(work(), "main") is None
    assert guide.nudge(work(step()), "main") == KICKOFF_NUDGE


def test_long_silence_asks_for_a_concrete_update():
    guide, clock = middleware()
    guide.nudge(work(), "main")
    clock.now = 10
    assert guide.nudge(work(step(index=0), step(index=1)), "main") is None
    clock.now = 25
    assert guide.nudge(work(step(index=0), step(index=1), step(index=2)), "main") == PROGRESS_NUDGE


def test_speaking_resets_the_silence():
    guide, clock = middleware()
    guide.nudge(work(), "main")
    clock.now = 30
    spoke = work(step(index=0), step("speak_to_user", 1))
    assert guide.nudge(spoke, "main") is None
    clock.now = 35
    assert guide.nudge([*spoke, *step(index=2)], "main") is None


def test_many_quick_silent_steps_also_ask_for_an_update():
    guide, _ = middleware()
    guide.nudge(work(), "main")
    assert guide.nudge(work(*(step(index=i) for i in range(6))), "main") == PROGRESS_NUDGE


def test_background_tasks_are_nudged_less_often():
    guide, clock = middleware()
    guide.nudge(work(), "main-task-abc")
    assert guide.nudge(work(step()), "main-task-abc") is None  # no kickoff for tasks
    clock.now = 30
    assert guide.nudge(work(step(index=0), step(index=1)), "main-task-abc") is None
    clock.now = 61
    assert guide.nudge(work(step(index=0), step(index=1), step(index=2)), "main-task-abc") == PROGRESS_NUDGE
