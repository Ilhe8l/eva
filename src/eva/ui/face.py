"""Eva's face: a small pixel-art head drawn with half-block characters.

Each terminal cell holds two square pixels, the top one as the foreground of
`▀` and the bottom one as its background. Every expression is an animation:
a function from the seconds since it began to a `Frame`. The animator layers
breathing, blinking and glances on top of the calm ones.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, replace

from rich.style import Style
from rich.text import Text

WIDTH, HEIGHT = 50, 22  # pixels; the face takes WIDTH cells by HEIGHT / 2 rows
HEAD = (23.0, 10.6, 16.6, 10.3)  # centre x, centre y, radius x, radius y; room on the right for a plant
VISOR = (23.0, 13.3, 12.9, 6.9)
EYE_CENTRES = (17.1, 28.9)
EYE_Y = 13.3

Colour = tuple[int, int, int]
VISOR_COLOUR: Colour = (8, 10, 15)
EYE_COLOUR: Colour = (64, 170, 255)
SPARKLE: Colour = (255, 214, 102)
ALERT_COLOUR: Colour = (255, 166, 64)
GLOW: Colour = (14, 44, 80)
POT_RIM: Colour = (205, 133, 90)
POT: Colour = (165, 98, 62)
STEM: Colour = (78, 160, 82)
LEAF: Colour = (118, 204, 98)
NEW_LEAF: Colour = (160, 228, 120)


@dataclass(frozen=True)
class Eye:
    """One eye in pixels.

    `shape` is "oval" or "arc" (only the upper rim: a smiling eye).
    `tilt` raises the outer corner (degrees). `top` and `bottom` trim an oval
    with a line `v = level + slope * outward`, in units of the radius, where
    `outward` grows towards the edge of the face.
    """

    rx: float = 3.5
    ry: float = 2.4
    tilt: float = 7.0
    top: tuple[float, float] | None = None
    bottom: tuple[float, float] | None = None
    shape: str = "oval"
    colour: Colour = EYE_COLOUR


CLOSED = Eye(rx=3.4, ry=0.55, tilt=0)
Spark = tuple[float, float, Colour]  # x, y relative to the visor centre
Pixel = tuple[int, int, Colour]  # x, y on the canvas


@dataclass(frozen=True)
class Frame:
    """One moment of an expression. An eye set to None is not drawn."""

    left: Eye | None
    right: Eye | None
    gaze: tuple[float, float] = (0.0, 0.0)  # pixels the eyes look towards
    bob: float = 0.0  # pixels the head moves down
    glow: float = 1.0
    sparks: tuple[Spark, ...] = ()  # move with the head
    scenery: tuple[Pixel, ...] = ()  # beside the head, still while it bobs
    calm: bool = True  # breathing, blinking and glances apply


def _both(eye: Eye, **frame) -> Frame:
    return Frame(eye, eye, **frame)


def _wave(t: float, hertz: float) -> float:
    return math.sin(2 * math.pi * hertz * t)


def neutral(t: float) -> Frame:
    return _both(Eye())


def happy(t: float) -> Frame:
    hop = -abs(_wave(t, 1.6)) * 1.4 if t < 1.3 else 0.0  # a little jump of joy, then content
    return _both(Eye(rx=3.6, ry=2.8, tilt=0, shape="arc"), gaze=(0, 0.6), bob=hop)


def amused(t: float) -> Frame:
    shake = 0.7 if _wave(t, 5) > 0 else -0.3  # laughing
    return _both(Eye(rx=3.8, ry=2.0, tilt=0, shape="arc"), gaze=(0, 0.4 + shake), bob=shake * 0.8, calm=False)


def proud(t: float) -> Frame:
    """Chin up, a satisfied smile, and sparkles twinkling around her head."""
    sparkles = []
    for (x, y), phase in (((6, 4), 0.0), ((41, 2), 0.45), ((44, 9), 0.8)):
        stage = int((t + phase) * 3) % 4  # off, a dot, a small star, a dot
        if stage:
            sparkles.append((x, y, SPARKLE))
        if stage == 2:
            sparkles += [(x + dx, y + dy, SPARKLE) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    smile = Eye(rx=3.6, ry=2.6, tilt=0, shape="arc")
    return _both(smile, gaze=(0, -0.4), bob=-1 if t < 0.6 else 0, scenery=tuple(sparkles))


def sad(t: float) -> Frame:
    eye = Eye(rx=3.3, ry=2.2, tilt=-14, top=(0.05, 0.9))
    fall = (t % 2.2) / 2.2  # a tear runs down from the right eye
    tear = (EYE_CENTRES[1] - VISOR[0] + 1.5, EYE_Y - VISOR[1] + 2.6 + fall * 3.2, EYE_COLOUR)
    return _both(eye, gaze=(0, 1.2), bob=1, sparks=(tear,) if fall < 0.8 else ())


def worried(t: float) -> Frame:
    glance = 1.3 if _wave(t, 0.9) > 0 else -1.3  # looking around nervously
    return _both(Eye(ry=2.6, tilt=-4, top=(-0.6, 0.45)), gaze=(glance, 0), calm=False)


def surprised(t: float) -> Frame:
    pop = 1.0 + 0.35 * math.exp(-t * 4) * math.cos(t * 12)  # pops open and settles
    return _both(Eye(rx=3.5 * pop, ry=3.2 * pop, tilt=0), bob=-1 if t < 0.35 else 0, calm=False)


def curious(t: float) -> Frame:
    lean = _wave(t, 0.35)
    big, small = Eye(rx=3.6, ry=2.9, tilt=4), Eye(rx=3.3, ry=1.4, tilt=-8)
    left, right = (big, small) if lean > 0 else (small, big)
    return Frame(left, right, gaze=(1.2 * lean, -0.4))


def confused(t: float) -> Frame:
    swap = _wave(t, 0.8) > 0
    big, small = Eye(rx=3.4, ry=2.7, tilt=-6), Eye(rx=3.0, ry=1.2, tilt=10)
    left, right = (big, small) if swap else (small, big)
    return Frame(left, right, gaze=(0.6 * _wave(t, 1.6), 0), calm=False)


def thinking(t: float) -> Frame:
    """A seedling grows in a pot beside her while she thinks, and she watches it."""
    return _both(Eye(ry=2.0, tilt=5), gaze=(1.7 + 0.4 * _wave(t, 0.3), 0.7), scenery=plant(t))


def plant(t: float) -> tuple[Pixel, ...]:
    pixels = [(x, 19, POT_RIM) for x in range(42, 49)]
    pixels += [(x, 20, POT) for x in range(43, 48)] + [(x, 21, POT) for x in range(44, 47)]
    stem = min(10, int(t / 0.3))  # one pixel every 0.3 s, up to y = 9
    pixels += [(45, 18 - i, STEM) for i in range(stem)]
    if stem >= 3:
        pixels += [(44, 16, LEAF), (43, 15, LEAF), (42, 15, LEAF)]
    if stem >= 6:
        pixels += [(46, 13, LEAF), (47, 12, LEAF), (48, 12, LEAF)]
    if stem >= 10:
        flutter = int(t / 0.7) % 2 if t > 3.6 else 0  # fully grown: the top leaves flutter
        pixels += [(44, 8, NEW_LEAF), (43, 7 + flutter, NEW_LEAF), (46, 8, NEW_LEAF), (47, 8 - flutter, NEW_LEAF)]
    return tuple(pixels)


def focused(t: float) -> Frame:
    reading = 1.3 * _wave(t, 0.45)  # scanning, like reading a line
    return _both(Eye(ry=2.2, tilt=-2, top=(-0.2, -0.55)), gaze=(reading, 0.6))


def sleepy(t: float) -> Frame:
    cycle = t % 4.0
    lid = 0.1 + 0.55 * min(cycle / 3.2, 1.0)  # the eyes slowly close, then jerk open
    rise = (t % 2.5) / 2.5
    z = (10.5 + rise * 4, -9.5 - rise * 3, (120, 150, 180))
    return _both(Eye(ry=2.3, tilt=0, top=(lid, 0.0)), gaze=(0, 0.8), bob=0.6, sparks=(z,), calm=False)


def wink(t: float) -> Frame:
    closed = 0.25 < t % 3.0 < 1.1
    right = Eye(rx=3.5, ry=2.2, tilt=0, shape="arc") if closed else Eye()
    return Frame(Eye(), right, bob=-0.5 if closed else 0.0)


def listening(t: float) -> Frame:
    size = 1.0 + 0.08 * _wave(t, 0.7)
    return _both(Eye(rx=3.5 * size, ry=2.7 * size, tilt=3), glow=0.85 + 0.15 * _wave(t, 0.7), gaze=(0, -0.3))


def loading(t: float) -> Frame:
    head = int(t * 14) % 16  # a spinner of 16 dots with a fading tail
    sparks = []
    for i in range(16):
        age = (head - i) % 16
        a = 2 * math.pi * i / 16
        colour = _mix(VISOR_COLOUR, EYE_COLOUR, max(0.15, 1.0 - age / 9))
        sparks.append((math.cos(a) * 5.5, math.sin(a) * 3.6, colour))
    return Frame(None, None, sparks=tuple(sparks), calm=False)


def glitch(t: float) -> Frame:
    tick = int(t * 12)
    jitter = random.Random(tick).uniform(-1.5, 1.5)
    dark = tick % 7 == 0
    eye = Eye(rx=3.2, ry=2.2, tilt=0, colour=ALERT_COLOUR)
    return _both(eye, gaze=(jitter, 0), glow=0.25 if dark else 1.0, calm=False)


ANIMATIONS: dict[str, Callable[[float], Frame]] = {
    "neutral": neutral,
    "happy": happy,
    "amused": amused,
    "proud": proud,
    "sad": sad,
    "worried": worried,
    "surprised": surprised,
    "curious": curious,
    "confused": confused,
    "thinking": thinking,
    "focused": focused,
    "sleepy": sleepy,
    "wink": wink,
    "listening": listening,
    "loading": loading,
    "glitch": glitch,
}


def draw(frame: Frame) -> list[list[Colour | None]]:
    """The frame as rows of pixels; None is transparent."""
    bob = round(frame.bob)
    gaze_x, gaze_y = frame.gaze
    eyes = [
        (EYE_CENTRES[0] + gaze_x, EYE_Y + bob + gaze_y, frame.left, -1),
        (EYE_CENTRES[1] + gaze_x, EYE_Y + bob + gaze_y, frame.right, 1),
    ]
    eyes = [eye for eye in eyes if eye[2] is not None]
    sparks = {(math.floor(VISOR[0] + x), math.floor(VISOR[1] + bob + y)): colour for x, y, colour in frame.sparks}
    sparks.update({(x, y): colour for x, y, colour in frame.scenery})
    rows = []
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            pixel = _pixel(x + 0.5, y + 0.5, bob, eyes, frame.glow)
            spark = sparks.get((x, y))
            if spark is not None and (pixel is None or in_visor(x + 0.5, y + 0.5, bob)):
                pixel = spark
            row.append(pixel)
        rows.append(row)
    return rows


def in_visor(x: float, y: float, bob: int) -> bool:
    vx, vy, vrx, vry = VISOR
    return ((x - vx) / vrx) ** 2 + ((y - vy - bob) / vry) ** 2 <= 1


def _pixel(x: float, y: float, bob: int, eyes, glow_level: float) -> Colour | None:
    hx, hy, hrx, hry = HEAD
    hy += bob
    head = ((x - hx) / hrx) ** 2 + ((y - hy) / hry) ** 2
    if head > 1:
        return None
    if not in_visor(x, y, bob):
        return _shell(x - hx, y - hy, head)
    glow = 0.0
    glow_colour = GLOW
    for ex, ey, eye, side in eyes:
        inside, distance = _eye(x - ex, y - ey, eye, side)
        if inside:
            return _mix(VISOR_COLOUR, eye.colour, max(0.0, min(1.0, glow_level)))
        if eye.shape == "oval":
            glow = max(glow, 1.0 - (distance - 1.0) / 0.45)
            glow_colour = _mix(VISOR_COLOUR, eye.colour, 0.3)
    if glow > 0:
        return _mix(VISOR_COLOUR, glow_colour, min(glow, 1.0) * glow_level)
    return VISOR_COLOUR


def _eye(dx: float, dy: float, eye: Eye, side: int) -> tuple[bool, float]:
    """Whether a pixel offset from the eye centre is lit, and its elliptic distance."""
    angle = math.radians(eye.tilt) * side
    u = dx * math.cos(angle) + dy * math.sin(angle)
    v = -dx * math.sin(angle) + dy * math.cos(angle)
    un, vn = u / eye.rx, v / eye.ry
    distance = math.hypot(un, vn)
    if eye.shape == "arc":
        return (0.58 <= distance <= 1.08 and vn < 0.25), distance
    if distance > 1:
        return False, distance
    outward = un * side
    trimmed = 1.4  # trimmed pixels barely glow, so the cut stays crisp
    if eye.top and vn < eye.top[0] + eye.top[1] * outward:
        return False, trimmed
    if eye.bottom and vn > eye.bottom[0] + eye.bottom[1] * outward:
        return False, trimmed
    return True, distance


def _shell(dx: float, dy: float, edge: float) -> Colour:
    """The white shell, lit from the upper left and darker towards the rim."""
    light = 0.72 - 0.22 * dx / HEAD[2] - 0.35 * dy / HEAD[3] - 0.25 * max(0.0, edge - 0.55)
    return _mix((150, 160, 172), (255, 255, 255), max(0.0, min(1.0, light)))


def _mix(a: Colour, b: Colour, amount: float) -> Colour:
    return tuple(round(x + (y - x) * amount) for x, y in zip(a, b, strict=True))


def render(frame: Frame) -> Text:
    """The frame as half-block characters; transparent pixels stay blank."""
    return to_text(draw(frame))


def to_text(pixels: list[list[Colour | None]]) -> Text:
    text = Text()
    for top, bottom in zip(pixels[0::2], pixels[1::2], strict=True):
        for upper, lower in zip(top, bottom, strict=True):
            if upper is None and lower is None:
                text.append(" ")
            elif upper is None:
                text.append("▄", Style(color=_rgb(lower)))
            else:
                text.append("▀", Style(color=_rgb(upper), bgcolor=_rgb(lower) if lower else None))
        text.append("\n")
    text.rstrip()
    return text


def _rgb(colour: Colour) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


class FaceAnimator:
    """Decides what the face shows over time.

    `mood` is what Eva is doing (thinking, focused, loading, ...) and lasts
    until it changes. A reaction plays over it for a few seconds, then the
    mood comes back. Calm expressions breathe, blink and glance around.
    """

    BREATH = 3.2  # seconds

    def __init__(self, clock: Callable[[], float], rng: random.Random | None = None) -> None:
        self._clock = clock
        self._rng = rng or random.Random()
        start = clock()
        self._mood = ("neutral", start)
        self._reaction: tuple[str, float, float] | None = None  # name, start, end
        self.speaking_until = 0.0
        self._next_blink = start + self._rng.uniform(2.0, 4.0)
        self._blink_until = 0.0
        self._next_glance = start + self._rng.uniform(5.0, 9.0)
        self._glance_until = 0.0
        self._glance = 0.0

    @property
    def mood(self) -> str:
        return self._mood[0]

    @mood.setter
    def mood(self, name: str) -> None:
        if name in ANIMATIONS and name != self._mood[0]:
            self._mood = (name, self._clock())

    def react(self, name: str, seconds: float = 4.0) -> None:
        if name in ANIMATIONS:
            now = self._clock()
            self._reaction = (name, now, now + seconds)

    def speak(self, seconds: float) -> None:
        self.speaking_until = max(self.speaking_until, self._clock() + seconds)

    @property
    def expression(self) -> tuple[str, float]:
        """The expression showing now, and when it began."""
        now = self._clock()
        if self._reaction and now < self._reaction[2]:
            return self._reaction[0], self._reaction[1]
        self._reaction = None
        return self._mood

    def frame(self) -> Frame:
        now = self._clock()
        name, since = self.expression
        frame = ANIMATIONS[name](now - since)
        if now < self.speaking_until:
            frame = _speaking(frame, now)
        if not frame.calm:
            return frame
        bob = frame.bob + (1 if _wave(now, 1 / self.BREATH) > 0.35 else 0)
        frame = replace(frame, bob=bob)
        if now >= self._next_blink:
            self._blink_until = now + 0.14
            double = self._rng.random() < 0.2
            self._next_blink = now + (0.3 if double else self._rng.uniform(2.5, 5.5))
        if now < self._blink_until:
            frame = replace(frame, left=_blink(frame.left), right=_blink(frame.right))
        if name == "neutral":
            if now >= self._next_glance:
                self._glance = self._rng.choice((-1.3, 1.3))
                self._glance_until = now + self._rng.uniform(0.9, 1.6)
                self._next_glance = now + self._rng.uniform(6.0, 11.0)
            if now < self._glance_until:
                frame = replace(frame, gaze=(frame.gaze[0] + self._glance, frame.gaze[1]))
        return frame


def _blink(eye: Eye | None) -> Eye | None:
    return CLOSED if eye is not None and eye.shape == "oval" else eye


def _speaking(frame: Frame, now: float) -> Frame:
    """Eyes squash a little with each syllable and glow brighter."""
    syllable = 0.82 + 0.18 * abs(_wave(now, 3.3))

    def squash(eye: Eye | None) -> Eye | None:
        return replace(eye, ry=eye.ry * syllable) if eye is not None and eye.shape == "oval" else eye

    return replace(frame, left=squash(frame.left), right=squash(frame.right), glow=0.85 + 0.15 * syllable)
