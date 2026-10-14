# Roadmap

What is planned for DIANA, roughly in order. Want to help with something? Open an
issue saying so first, so that work is not duplicated; items marked
**good first contribution** need little background in the code base.

Status: ✅ done · 🔶 in progress · ⬜ not started

## Done

- ✅ Dual-modality pipeline (materials → phantoms → projection → artifacts →
  reconstruction → bimodal histogram → metrics), 7 phantom presets, 9
  reconstruction algorithms, segmented-volume import, neutron spectra and Bragg-edge
  scans with NCrystal (v1.x, Apr–Jun 2026).
- ✅ Cone-beam / translated laminography geometry, NMC particle phantoms, density
  sweeps and tomography–laminography Fourier fusion (Aug 2026).
- ✅ v2.0.0: package split into subpackages, CPU projector and detection-chain fixes,
  tutorials, docs (Sep 2026).
- ✅ Collaboration set-up: CI on every push (Linux / macOS / Windows, Python
  3.9–3.13, oldest dependencies), coverage gate, numerical regression guard, release
  workflow, pre-commit hooks, citation file (Oct 2026).

## Next: correctness

- ⬜ **Fix the known bugs recorded as expected failures** (`pytest -rx` lists them,
  with file and line). Each fix removes one `xfail` marker. Several are
  **good first contributions** (algorithm aliases, error messages, input
  validation, plots that drop series beyond the 4th–6th). The ones that change
  simulated numbers (ring-removal default, TV-minimisation step size, histogram panel
  extent, wavelength-domain spectrum plot) need a CHANGELOG entry and a regenerated
  regression reference.
- ⬜ Re-validate the published figures in `results/` against v2.0 numerics.
- ⬜ Add the missing NIST XCOM X-ray tables (Ca, N, Si, …) so materials containing
  them can be built — **good first contribution**.
- ⬜ GPU test runner (self-hosted or scheduled) for the ASTRA-only code paths:
  iterative reconstruction, cone-beam projection and laminography.

## Next: physics and models

- ⬜ Wavelength-dependent coherent scatter (Bragg edges) in the base neutron model,
  not only in the Bragg-edge scan.
- ⬜ Detector halo and background-floor model (calibrated detector response) in
  `acquisition`, and layered battery-electrode phantoms with transport-driven
  composition.

## Next: 5-D (3-D × time × wavelength × modality) imaging

New modules for time-of-flight neutron imaging (e.g. ODIN at ESS) combined with
X-rays. Each lands on its own branch with tests from the first commit.

| Module | Purpose | Status |
|---|---|---|
| `phantoms/fields.py` | Time- and space-varying phase / composition fields | ⬜ |
| `acquisition/tof.py` | Time-of-flight neutron acquisition | ⬜ |
| `acquisition/xray_coupled.py` | Coupled perpendicular X-ray acquisition | ⬜ |
| `reconstruction/spectral.py` | Count-domain spectral reconstruction with an X-ray constraint and TV | ⬜ |

## Infrastructure

- ⬜ First tagged release (`v2.0.0`) archived on Zenodo for a citable software DOI.
- ⬜ Rendered documentation site (GitHub Pages) from `docs/`.
- ⬜ PyPI package (the release workflow is ready; needs a trusted-publisher set-up).
