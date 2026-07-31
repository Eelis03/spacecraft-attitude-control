"""Tier three: every example script runs to completion under reduced settings.

The scripts are executed as subprocesses, the way a reader of the README would run
them, so a broken import or a missing command line option is caught here rather
than in the documentation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


def test_every_example_is_collected() -> None:
    """The scripts below are the ones the README documents.

    The runnable tests discover scripts from the directory, so a new example is
    covered automatically. This check exists for the other direction: it fails if
    a documented script is renamed or removed without the README following.
    """
    assert [path.name for path in EXAMPLES] == [
        "attitude_representations.py",
        "disturbance_rejection.py",
        "momentum_dumping.py",
        "slew_manoeuvre.py",
    ]


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs(script: Path, tmp_path: Path) -> None:
    """Each script exits cleanly and prints something under ``--quick``.

    Figures are suppressed and the working directory is a temporary one, so the
    test writes nothing into the repository.
    """
    result = subprocess.run(
        [sys.executable, str(script), "--quick", "--no-figure"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.stem)
def test_example_accepts_help(script: Path, tmp_path: Path) -> None:
    """Each script documents its own options."""
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--quick" in result.stdout
