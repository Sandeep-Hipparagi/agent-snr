#!/usr/bin/env python3
"""Fresh virtual environment smoke test verifying public installation of agent-snr."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Execute a subprocess command, streaming output and raising on failure."""
    print(f"\n[bold cyan]==>[/] [bold]{description}...[/]")
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed during: {description}")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        sys.exit(result.returncode)
    if result.stdout.strip():
        lines = result.stdout.strip().splitlines()
        for line in lines[-5:]:
            print(f"    {line}")
    print(f"    [OK] {description} succeeded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify clean venv installation and smoke execution of agent-snr."
    )
    parser.add_argument(
        "--pkg",
        default="agent-snr",
        help=(
            "Package specifier to install "
            "(e.g. 'agent-snr', 'agent-snr==0.1.1', or local wheel path)."
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Build and test the latest local wheel from dist/ instead of remote PyPI index.",
    )
    args = parser.parse_args()

    pkg_target = args.pkg
    if args.local:
        dist_dir = Path(__file__).resolve().parent.parent / "dist"
        wheels = sorted(dist_dir.glob("*.whl"), key=os.path.getmtime, reverse=True)
        if not wheels:
            print(f"[ERROR] No wheels found in {dist_dir}. Run 'python -m build' first.")
            sys.exit(1)
        pkg_target = str(wheels[0])
        print(f"Testing latest local wheel: {pkg_target}")

    with tempfile.TemporaryDirectory(prefix="agent_snr_verify_") as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"Creating isolated temporary virtual environment in {tmp_path}...")
        venv.create(tmp_path, with_pip=True)

        if sys.platform == "win32":
            py_bin = tmp_path / "Scripts" / "python.exe"
            cli_bin = tmp_path / "Scripts" / "agent-snr.exe"
        else:
            py_bin = tmp_path / "bin" / "python"
            cli_bin = tmp_path / "bin" / "agent-snr"

        # 1. Install package using python -m pip to prevent file-locking on Windows
        run_command(
            [str(py_bin), "-m", "pip", "install", "--upgrade", "pip"],
            "Upgrading pip in isolated venv",
        )
        run_command(
            [str(py_bin), "-m", "pip", "install", pkg_target],
            f"Installing '{pkg_target}' in isolated venv",
        )

        # 2. Test Public Imports
        test_import_code = (
            "import agent_snr\n"
            "from agent_snr import (\n"
            "    AgentSNRFirewall,\n"
            "    SNRWrappedOpenAI,\n"
            "    LangGraphSNRAdapter,\n"
            "    CrewAISNRHook,\n"
            "    attach_to_otel_span,\n"
            ")\n"
            "print(f'Successfully imported agent-snr version {agent_snr.__version__}')\n"
        )
        run_command(
            [str(py_bin), "-c", test_import_code],
            "Verifying public class imports and __version__",
        )

        # 3. Test CLI Entrypoint
        run_command(
            [str(cli_bin), "demo"],
            "Running 'agent-snr demo' CLI diagnostics",
        )

    print("\n[SUCCESS] End-to-end clean venv smoke test passed successfully!\n")


if __name__ == "__main__":
    main()
