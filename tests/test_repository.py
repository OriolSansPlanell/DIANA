"""Repository consistency: version numbers, documentation links, notebook APIs.

These tests catch the drift that creeps in when several people edit docs,
notebooks and code independently: a renamed function that a research notebook
still imports, a moved docs page, a version bump that missed the citation file.
"""
import ast
import importlib
import json
import re
from pathlib import Path

import pytest

import neutron_xray_sim as nxs

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted(
    (ROOT / "tutorials").glob("*.ipynb"))
SCRIPTS = sorted((ROOT / "examples").glob("*.py"))
MARKDOWN = sorted(p for p in ROOT.rglob("*.md")
                  if not any(part.startswith(".") or part in {"build", "dist"}
                             for part in p.relative_to(ROOT).parts[:-1]))


# ── version ──────────────────────────────────────────────────────────────────

def test_version_matches_changelog_and_citation():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.findall(r"^## \[(\d+\.\d+\.\d+[^\]]*)\]", changelog, flags=re.M)
    assert released, "CHANGELOG.md has no '## [x.y.z]' release heading"
    assert nxs.__version__ == released[0], (
        f"_version.py says {nxs.__version__} but the latest CHANGELOG release is "
        f"{released[0]}")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r"^version:\s*\"?([^\"\n]+)\"?\s*$", citation, flags=re.M)
    assert m and m.group(1) == nxs.__version__, "update 'version:' in CITATION.cff"


def test_changelog_has_an_unreleased_section():
    # Contributors add their entries under [Unreleased]; the release workflow
    # turns it into the next version.
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## \[Unreleased\]", changelog, flags=re.M)


# ── documentation links ──────────────────────────────────────────────────────

_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")


def _relative_links(md: Path):
    text = md.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)       # ignore code blocks
    for a, b in _LINK.findall(text):
        target = a or b
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue                                         # URL, mailto:, anchor
        yield target.split("#")[0]


@pytest.mark.parametrize("md", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(md):
    missing = [t for t in _relative_links(md) if t and not (md.parent / t).exists()]
    assert not missing, f"broken links in {md.relative_to(ROOT)}: {missing}"


# ── notebooks and examples use names that exist ──────────────────────────────

def _code(path: Path) -> str:
    if path.suffix == ".py":
        return path.read_text(encoding="utf-8")
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Drop IPython magics and shell escapes, which are not Python.
        src = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                        for line in src.splitlines())
        cells.append(src)
    return "\n".join(cells)


def _api_references(tree: ast.AST):
    """Yield (module, name) pairs the code expects to exist in the package."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "neutron_xray_sim":
                    yield a.name, None
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] == "neutron_xray_sim":
                for a in node.names:
                    if a.name != "*":
                        yield node.module, a.name
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            yield aliases[node.value.id], node.attr


@pytest.mark.parametrize("path", NOTEBOOKS + SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_uses_existing_api(path):
    try:
        tree = ast.parse(_code(path))
    except SyntaxError as exc:
        pytest.fail(f"{path.name} does not parse as Python: {exc}")
    missing = set()
    for module, name in _api_references(tree):
        try:
            mod = importlib.import_module(module)
        except ImportError:
            # `from neutron_xray_sim import gui` style: a submodule, not an attribute.
            missing.add(module)
            continue
        if name is not None and not hasattr(mod, name):
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.add(f"{module}.{name}")
    assert not missing, f"{path.relative_to(ROOT)} uses names that no longer exist: " \
                        f"{sorted(missing)}"


# ── optional GUI ─────────────────────────────────────────────────────────────

def test_gui_module_imports():
    pytest.importorskip("PyQt5")
    from neutron_xray_sim.gui import app
    assert callable(app.main)
