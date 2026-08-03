"""What the distribution and the repository promise to anything downstream.

These are not physics tests. They cover two claims that are easy to make and easy
to break silently: that the package delivers its type information to code that
installs it, and that the figures the README shows are actually in the repository
and small enough to keep cloning cheap.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import attitude_control

PACKAGE_DIR = Path(attitude_control.__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "docs" / "figures"

# The published figures share this budget, which is also enforced by
# ``scripts/publish_figures.py`` when it regenerates them.
FIGURE_BUDGET_BYTES = 250 * 1024

# Alt text has to say what the figure shows. A caption shorter than this is a
# label rather than a description.
MINIMUM_ALT_TEXT = 20

_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)\)")


def test_the_package_ships_a_py_typed_marker() -> None:
    """PEP 561: without this file the type hints are invisible to an installer.

    The package is checked by mypy in strict mode, which is worth nothing to a
    downstream project unless the marker is present in the installed package. The
    directory tested here is the imported one, so a packaging configuration that
    dropped the file from the wheel would fail this.
    """
    marker = PACKAGE_DIR / "py.typed"
    assert marker.is_file(), f"{marker} is missing"
    assert marker.read_bytes() == b"", "the marker is a flag, not a configuration file"


def test_published_figures_exist_and_fit_the_budget() -> None:
    """Every tracked figure is present, non-empty, and inside the shared budget."""
    figures = sorted(FIGURE_DIR.glob("*.png"))
    assert figures, f"no published figure in {FIGURE_DIR}"
    for path in figures:
        assert path.stat().st_size > 0
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    total = sum(path.stat().st_size for path in figures)
    assert total <= FIGURE_BUDGET_BYTES, f"tracked figures total {total} bytes"


@pytest.mark.parametrize("document", ["README.md", "docs/design-notes.md"])
def test_every_embedded_figure_resolves_and_describes_itself(document: str) -> None:
    """A figure reference points at a file that exists and carries real alt text.

    The alt text is the only description a reader gets when the image does not
    load, and it is what a screen reader announces, so a length check is a crude
    but effective guard against captions like "figure 1".
    """
    path = REPO_ROOT / document
    for match in _IMAGE.finditer(path.read_text(encoding="utf-8")):
        target, alt = match.group("target"), match.group("alt").strip()
        if target.startswith(("http://", "https://")):
            continue
        assert (path.parent / target).is_file(), f"{document} points at missing {target}"
        assert len(alt) >= MINIMUM_ALT_TEXT, f"weak alt text in {document}: {alt!r}"
