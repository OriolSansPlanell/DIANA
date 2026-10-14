# Contributing to DIANA

Thanks for helping! This page explains how the code is organised, how to set up a
development environment, and the few conventions that keep collaboration smooth.
Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Planned work is in [ROADMAP.md](ROADMAP.md); known bugs are listed by `pytest -rx`.

## 1. Set up

```bash
git clone https://github.com/OriolSansPlanell/DIANA.git
cd DIANA
python -m venv .venv && source .venv/bin/activate      # or a conda env (needed for ASTRA)
pip install -e ".[dev,notebooks]"
pre-commit install                                     # installs the commit and push hooks
pytest                                                 # ~1 min on CPU
```

`pre-commit install` sets up two git hooks: on **commit** it strips notebook outputs,
runs ruff and fixes whitespace; on **push** it runs the fast tests
(`pytest -m "not slow"`), so a broken commit is caught before it reaches GitHub.
`git push --no-verify` skips them in an emergency — CI runs the same checks anyway.

Everything runs without a GPU. If you have one, `conda install -c astra-toolbox -c nvidia
astra-toolbox` enables the ASTRA backend; the tests use the CPU backends.

## 2. Where does my code go?

The package is split by pipeline stage. Each subpackage has one job and depends only on the
ones above it in this list:

| Subpackage | Put here… | Examples |
|---|---|---|
| `physics/` | element data, materials, beam spectra | a new element, a material, a neutron source model |
| `phantoms/` | ground-truth volumes | a new preset, an importer for a new file format |
| `acquisition/` | anything that turns a phantom into measured sinograms | a projector, a new artifact, a noise model |
| `reconstruction/` | sinograms → volumes | a reconstruction algorithm, volume fusion |
| `analysis/` | volumes → numbers | histogram metrics, segmentation, signatures |
| `plotting/` | figures (never computes anything expensive) | a new plot |
| `simulation.py` | orchestration of the stages | new `DualModalitySimulation` options |
| `io.py` | persistence | new cached artifacts |

Recipes for the most common additions:

* **Element** — add `ATOMIC_MASS` and `NEUTRON_XS` entries in `physics/elements.py`, and the
  NIST XCOM export as `physics/data/xray/<El>.txt`.
* **Material** — `register_material("key", material_from_formula(...))` in your own code, or
  add it to `MATERIALS` in `physics/materials.py` if it is generally useful.
* **Phantom preset** — write `make_<name>_phantom(N=64, voxel_cm=None, Nx=None, Ny=None,
  Nz=None)` in `phantoms/presets.py` using `resolve_grid` + `PhantomBuilder`, and add it to
  `PHANTOM_PRESETS`.
* **Artifact** — add the fields to `ArtifactConfig` (default *off*), implement a private
  `_apply_<name>` function in `acquisition/artifacts.py`, and call it at the right place in
  the detection chain inside `inject_sinogram_artifacts` / `inject_volume_artifacts`.
* **Reconstruction algorithm** — add it to `AVAILABLE_ALGORITHMS` and the dispatch in
  `reconstruction/reconstructor.py`; provide a CPU fallback or a clear warning.
* **Metric** — build on the shared helpers in `analysis/quality.py`
  (`ground_truth_positions`, `match_components`, `davies_bouldin`, `pairwise_overlap`)
  rather than re-implementing them.

If something should be importable as `neutron_xray_sim.<name>`, export it from the
subpackage `__init__.py` **and** the top-level `__init__.py` (`__all__`).

## 3. Conventions

* **Units:** linear attenuation in cm⁻¹, lengths in cm (voxel sizes `voxel_cm`), X-ray
  energies in keV, neutron energies in meV.
* **Array order:** volumes are `(Nz, Nx, Ny)`; coordinates `(z, x, y)` from the volume
  centre; sinograms are `(n_angles, n_slices, n_detector)`.
* **No global side effects on import:** no `plt.style.use`, no directories created, no
  warnings. Functions that save files take an output-directory argument.
* **Randomness** goes through an explicit `numpy.random.Generator` or a seed argument.
* **Optional dependencies** (ASTRA, TomoPy, NCrystal, tifffile, PyQt5) are imported inside
  the functions that need them, with a helpful error or a CPU fallback.
* **Docstrings** use the NumPy style already used in the code; state units.
* **Style:** follow the surrounding code. `ruff` checks for real problems (undefined
  names, unused imports, mutable defaults), not formatting.

## 4. Tests

* Every bug fix gets a regression test; every feature gets a test. Tests live in
  `tests/test_<module>.py`, mirroring the package.
* Keep tests fast: `N ≤ 48`, a few dozen angles, `use_astra=False`, `verbose=False`.
  Mark anything slower with `@pytest.mark.slow` (`pytest -m "not slow"` skips them).
* Shared fixtures live in `tests/conftest.py`.
* Tests run on CPU only. Code that needs ASTRA, TomoPy or NCrystal is tested with
  `pytest.importorskip(...)`, or with `monkeypatch` to exercise the fallback path.
* **Simulated numbers are guarded.** `tests/test_regression.py` recomputes a small
  set of values from every pipeline stage and compares them with
  `tests/data/regression_reference.json`. If your change alters them **on purpose**,
  regenerate the file with `pytest tests/test_regression.py --update-reference`,
  commit it, explain the change in the PR and add a CHANGELOG entry under
  "Changes that alter simulated numbers" — people compare against published results.
* **Known bugs are recorded as expected failures**: a test marked
  `@pytest.mark.xfail(strict=True, reason="BUG: …")` that describes the bug and where
  it is. `pytest -rx` lists them; they are a good place to start contributing. When
  you fix one, the test starts passing, which strict mode reports as a failure:
  remove the marker and the test becomes the regression test. Record a newly found
  bug the same way if you cannot fix it in your PR.
* Coverage: `pytest --cov` prints it; CI fails if it drops below 80 %.

## 5. Notebooks

* `tutorials/` — teaching notebooks. They must run top to bottom on a CPU in a few minutes
  (CI executes all of them). Keep them small, explained, and ending with an exercise.
* `notebooks/` — research / paper notebooks. They may need a GPU, large memory or data.
* `examples/` — non-interactive scripts.
* **Never commit outputs.** `pre-commit` runs `nbstripout` automatically; CI rejects
  notebooks with outputs. Put figures you want to keep in `results/` or a release.
* Don't copy helper functions between notebooks — if two notebooks need it, it belongs in
  the package.

## 6. Pull requests

1. Open or comment on an issue first for anything bigger than a small fix, so the
   design can be agreed before you write the code. `ROADMAP.md` lists planned work.
2. Branch from an up-to-date `main` (`git switch -c my-feature`); never commit to
   `main` directly. External contributors: fork, then branch.
3. Keep PRs focused; one topic per PR is much easier to review.
4. Update docs (`docs/`) and the `[Unreleased]` section of `CHANGELOG.md` for
   user-visible changes.
5. CI runs on **every push** to any branch and on every PR:

   | Check | What it does |
   |---|---|
   | Lint | `pre-commit run --all-files` (ruff, notebook outputs, whitespace, YAML/TOML) |
   | Tests | `pytest` on Linux (Python 3.9–3.13), macOS and Windows |
   | Oldest dependencies | `pytest` with the minimum versions in `pyproject.toml` |
   | Coverage | fails below 80 % |
   | Tutorials | executes every notebook in `tutorials/` |
   | Package | builds sdist + wheel, `twine check`, installs and imports the wheel |
   | All checks passed | the single status a PR needs before merging |

6. A code owner (see `.github/CODEOWNERS`) is asked for review automatically. Address
   comments with new commits rather than force-pushing, so reviewers can see what
   changed. Merge with **Squash and merge**: each PR becomes one commit on `main`
   whose message is the PR title, so make the title a good one-line summary.

## 7. Releases (maintainers)

1. Move the `[Unreleased]` entries under `## [X.Y.Z] — YYYY-MM-DD` in `CHANGELOG.md`.
2. Set the version in `neutron_xray_sim/_version.py` and `version:` /
   `date-released:` in `CITATION.cff` (a test checks they agree).
3. Merge to `main`, then tag: `git tag -a vX.Y.Z -m "DIANA X.Y.Z" && git push origin vX.Y.Z`.
4. The release workflow builds the package and publishes a GitHub Release with the
   changelog section; with the Zenodo integration on, the release gets a DOI.

Versions follow [semantic versioning](https://semver.org): a release that changes
simulated numbers or breaks an import path gets at least a minor version bump and is
announced at the top of its changelog entry.
