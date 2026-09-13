import os
import subprocess
from pathlib import Path


class HostCommandRunner:
    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    def run(self, command: str, cwd: str) -> str:
        directory = Path(cwd).expanduser().resolve()
        if not directory.is_dir():
            return f"Working directory does not exist: {directory}"
        try:
            result = subprocess.run(
                command,
                cwd=directory,
                shell=True,
                executable="/bin/bash",
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout} seconds"
        output = (result.stdout + result.stderr).strip()
        return f"Exit code: {result.returncode}\n{output[:12000]}"
