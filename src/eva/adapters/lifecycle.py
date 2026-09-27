"""Restarting Eva after she changes her own source."""

import subprocess
import sys
from pathlib import Path

TEST_TIMEOUT_SECONDS = 300
OUTPUT_TAIL_CHARACTERS = 6000


class SelfUpdater:
    """Runs the test suite and, when it passes, asks the terminal to restart."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.restart_requested = False

    def request_restart(self) -> str:
        try:
            tests = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Not restarting: tests timed out after {TEST_TIMEOUT_SECONDS} seconds."
        output = (tests.stdout + tests.stderr)[-OUTPUT_TAIL_CHARACTERS:]
        if tests.returncode != 0:
            return f"Not restarting: tests failed. Fix them first.\n{output}"
        self.restart_requested = True
        return f"Tests passed. Eva restarts with the new code when this turn ends.\n{output}"
