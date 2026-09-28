"""Desktop notifications, for when the user is not looking at the terminal."""

import shutil
import subprocess


class DesktopNotifier:
    """Uses `notify-send` (libnotify) when available; otherwise does nothing."""

    def __init__(self) -> None:
        self._command = shutil.which("notify-send")

    def notify(self, title: str, body: str = "") -> None:
        if self._command:
            subprocess.Popen(
                [self._command, "--app-name=Eva", title, body],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
