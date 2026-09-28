import random

import pytest

from eva.domain.models import MOODS
from eva.ui.face import ANIMATIONS, HEIGHT, WIDTH, FaceAnimator, draw, render


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def frames_over(animator, clock, seconds=6.0, fps=10):
    frames = []
    for _ in range(int(seconds * fps)):
        frames.append(draw(animator.frame()))
        clock.now += 1 / fps
    return frames


def test_every_mood_eva_can_choose_has_an_animation():
    assert set(MOODS) <= set(ANIMATIONS)


@pytest.mark.parametrize("name", list(ANIMATIONS))
def test_every_expression_moves(name):
    clock = Clock()
    animator = FaceAnimator(clock, random.Random(1))
    animator.mood = name
    frames = frames_over(animator, clock)
    assert all(len(frame) == HEIGHT and all(len(row) == WIDTH for row in frame) for frame in frames)
    distinct = {tuple(tuple(row) for row in frame) for frame in frames}
    assert len(distinct) > 1, f"{name} is a still image"


def test_a_reaction_plays_over_the_mood_and_then_fades_back():
    clock = Clock()
    animator = FaceAnimator(clock)
    animator.mood = "focused"
    animator.react("happy", seconds=2)
    assert animator.expression[0] == "happy"
    clock.now += 2.5
    assert animator.expression[0] == "focused"


def test_unknown_expressions_are_ignored():
    animator = FaceAnimator(Clock())
    animator.react("smug")
    animator.mood = "smug"
    assert animator.expression[0] == "neutral"


def test_the_face_fits_in_half_as_many_rows_as_pixels():
    lines = render(FaceAnimator(Clock()).frame()).plain.split("\n")
    assert len(lines) == HEIGHT // 2
    assert all(len(line) <= WIDTH for line in lines)


def test_a_plant_grows_while_eva_thinks():
    clock = Clock()
    animator = FaceAnimator(clock)
    animator.mood = "thinking"
    sprout = len(animator.frame().scenery)
    clock.now += 4
    assert len(animator.frame().scenery) > sprout
