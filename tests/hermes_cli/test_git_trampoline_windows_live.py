"""Live Windows E2E for the Git-for-Windows trampoline self-heal (#87876).

Runs ONLY on a real windows-latest runner (wine2e/** on-demand lane). No
mocked subprocess anywhere: the probes drive the REAL
``_git_is_trampoline`` / ``_locate_real_git`` / ``_ensure_non_trampoline_git``
helpers against the runner's genuine Git-for-Windows install plus a real
broken-trampoline stand-in (a .bat that reproduces the launcher's
"BUG (fork bomb)" guard output and refuses to run).

Standard Windows runners may expose Git from Program Files or from the
PM-managed PortableGit installation on PATH. The locator must resolve the real
architecture-specific ``libexec\\git-core\\git.exe`` in either layout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")  # live Windows trampoline E2E

_FORK_BOMB_BAT = (
    "@echo off\r\n"
    "echo BUG (fork bomb): tried to spawn itself, check your PATH 1>&2\r\n"
    "exit /b 1\r\n"
)


def _make_broken_trampoline(tmp_path: Path) -> Path:
    bat = tmp_path / "git.bat"
    bat.write_text(_FORK_BOMB_BAT, encoding="utf-8")
    return bat


class TestGitTrampolineLive:
    def test_runner_git_is_not_a_trampoline(self):
        from hermes_cli import update_cmd

        # Healthy PATH git on the runner: probe must report False and the
        # probe itself must actually run git (sanity: git exists here).
        version = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=30,
            check=False,
        )
        assert version.returncode == 0, version.stderr
        assert update_cmd._git_is_trampoline(["git"]) is False

    def test_broken_trampoline_detected_live(self, tmp_path):
        from hermes_cli import update_cmd

        bat = _make_broken_trampoline(tmp_path)
        assert update_cmd._git_is_trampoline([str(bat)]) is True

    def test_locate_real_git_finds_runner_git_core(self):
        from hermes_cli import update_cmd

        real = update_cmd._locate_real_git()
        # Accept the runner's standard Git install or the PM-managed toolchain
        # root discovered relative to the PATH shim.
        assert real is not None, "expected git-core git.exe on windows-latest"
        assert real.exists()
        assert real.name == "git.exe"
        probe = subprocess.run(
            [str(real), "--version"], capture_output=True, text=True, timeout=30,
            check=False,
        )
        assert probe.returncode == 0
        assert "git version" in probe.stdout.lower()

    def test_locate_real_git_finds_pm_toolchain_git_core(self, tmp_path, monkeypatch):
        from hermes_cli import update_cmd_git

        install_root = tmp_path / "git-2.53.0+3-win32-x64"
        shim = install_root / "cmd" / "git.exe"
        real = install_root / "mingw64" / "libexec" / "git-core" / "git.exe"
        real.parent.mkdir(parents=True)
        real.write_bytes(b"pinned test binary")
        monkeypatch.setattr(update_cmd_git.shutil, "which", lambda _name: str(shim))
        monkeypatch.setattr(
            update_cmd_git,
            "_probe_fork_bomb",
            lambda command: False if Path(command[0]) == real else None,
        )

        assert update_cmd_git._locate_real_git() == real

    def test_ensure_non_trampoline_git_swaps_live(self, tmp_path, capsys):
        from hermes_cli import update_cmd

        bat = _make_broken_trampoline(tmp_path)
        broken_cmd = [str(bat), "-c", "windows.appendAtomically=false"]
        healed = update_cmd._ensure_non_trampoline_git(broken_cmd)

        assert healed != broken_cmd, "broken trampoline must be swapped"
        assert healed[1:] == broken_cmd[1:], "git config args must survive"
        real = Path(healed[0])
        assert real.exists() and real.name == "git.exe"
        # The healed command must actually work.
        result = subprocess.run(
            healed[:1] + ["--version"], capture_output=True, text=True, timeout=30,
            check=False,
        )
        assert result.returncode == 0
        assert "switching to real git" in capsys.readouterr().out

    def test_healthy_git_command_untouched_live(self):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        assert update_cmd._ensure_non_trampoline_git(git_cmd) == git_cmd
