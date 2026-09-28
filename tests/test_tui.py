import asyncio
from types import SimpleNamespace

from eva.adapters.policy import ApprovalPolicy
from eva.adapters.voice import SpeechChannel
from eva.domain.models import ActionRequest, Reaction, Speech, TextDelta, ToolResult, ToolUse
from eva.ui.tui import ConversationLog, TuiView


class SilentPlayer:
    def __init__(self):
        self.played = []

    def play(self, text):
        self.played.append(text)


def text_of(log):
    return "\n".join("".join(segment.text for segment in strip) for strip in log.lines)


async def run(scenario):
    view = TuiView(SilentPlayer())
    app = SimpleNamespace(speech=SpeechChannel(), policy=ApprovalPolicy(), tasks=SimpleNamespace(running=list))
    view._terminal = SimpleNamespace(app=app, listener=SimpleNamespace(active=False))
    view._loop = asyncio.get_running_loop()
    view.console.bind(view._loop)
    async with view.app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        return await scenario(view, pilot)


def test_the_reply_goes_left_and_the_work_goes_right():
    async def scenario(view, pilot):
        view.user_said("check the weather")
        await asyncio.to_thread(view, ToolUse("write_file", {"file_path": "/skills/weather/SKILL.md"}))
        await asyncio.to_thread(view, ToolUse("execute", {"command": "curl wttr.in"}, call_id="1"))
        await asyncio.to_thread(view, ToolResult("execute", False, "HTTP 503", call_id="1"))
        await asyncio.to_thread(view, TextDelta("No **umbrella** needed.\n\nEnjoy the sun."))
        await asyncio.to_thread(view.end_turn)
        await pilot.pause(0.2)
        chat = text_of(view.app.query_one(ConversationLog))
        activity = text_of(view.app.query_one("#activity"))
        files = str(view.app.query_one("#files").render())
        return chat, activity, files, view.face.expression[0]

    chat, activity, files, expression = asyncio.run(run(scenario))
    assert "check the weather" in chat and "No umbrella needed." in chat and "Enjoy the sun." in chat
    assert "curl wttr.in" not in chat
    assert "$ curl wttr.in" in activity and "✗ HTTP 503" in activity
    assert "+ /skills/weather/SKILL.md" in files
    assert expression == "worried"


def test_one_key_answers_an_approval():
    async def scenario(view, pilot):
        action = ActionRequest("execute", {"command": "rm -rf build"})
        question = "Approve? [y]es / [N]o / [a]lways this session: "
        asking = asyncio.ensure_future(asyncio.to_thread(view.console.ask, question, "", action))

        async def answer_lines():
            while True:
                view.console.answer(await view.console.read())

        reader = asyncio.ensure_future(answer_lines())
        await pilot.pause(0.3)
        was_asking = view.app.query_one("#prompt-row").has_class("asking")
        await pilot.press("n")
        answer = await asyncio.wait_for(asking, timeout=2)
        await pilot.pause(0.1)
        reader.cancel()
        return was_asking, answer, text_of(view.app.query_one("#activity")), view.app.query_one("#prompt-row")

    was_asking, answer, activity, row = asyncio.run(run(scenario))
    assert was_asking and answer == "n"
    assert "rm -rf build" in activity and "declined" in activity
    assert not row.has_class("asking")


def test_eva_chooses_her_expressions_and_restarts_show_loading():
    async def scenario(view, pilot):
        await asyncio.to_thread(view, Reaction("proud"))
        reacted = view.face.expression[0]
        await asyncio.to_thread(view, Speech("Found it!", aloud=True, mood="happy"))
        spoken = view.face.expression[0]
        await asyncio.to_thread(view, ToolUse("restart_eva", {}))
        return reacted, spoken, view.face.mood, view.player.played

    reacted, spoken, mood, played = asyncio.run(run(scenario))
    assert (reacted, spoken, mood) == ("proud", "happy", "loading")
    assert played == ["Found it!"]


def test_stray_prints_land_in_the_conversation():
    async def scenario(view, pilot):
        print("Downloading the speech model...")
        await pilot.pause(0.2)
        return text_of(view.app.query_one(ConversationLog))

    assert "Downloading the speech model..." in asyncio.run(run(scenario))
