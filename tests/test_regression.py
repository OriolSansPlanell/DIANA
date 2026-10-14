"""Numerical regression guard: simulated numbers must not change by accident.

People compare DIANA output against published figures, so a change in any
simulated number has to be deliberate and announced (CONTRIBUTING.md §4). This
test recomputes a small, deterministic set of quantities that touch every
pipeline stage (material data → phantom → projection → detection chain →
reconstruction) and compares them with ``tests/data/regression_reference.json``.

If you changed the numbers **on purpose**, regenerate the reference and commit
it together with a CHANGELOG entry under "Changes that alter simulated numbers":

    pytest tests/test_regression.py --update-reference

Noise and GMM fits are left out on purpose: NumPy does not promise identical
random streams across versions, and scikit-learn's initialisation varies.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.physics.materials import MATERIALS, parse_formula
from neutron_xray_sim.physics.neutron_spectra import mu_n_for_beam

REFERENCE = Path(__file__).parent / "data" / "regression_reference.json"

# Relative tolerances per group. Physics tables are exact arithmetic; projection
# and FBP go through SciPy / scikit-image interpolation and filtering, which
# may differ in the last digits between library versions and platforms.
RTOL = {"materials": 1e-9, "beams": 1e-6, "phantom": 0.0, "sinograms": 1e-4,
        "detection_chain": 1e-4, "reconstruction": 1e-3}


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"sum": float(a.sum()), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def compute_values() -> dict:
    out = {}

    out["materials"] = {
        key: {"mu_n": float(m.mu_n),
              **{f"mu_x_{e}keV": float(m.mu_x_at(e)) for e in (40.0, 80.0, 150.0)}}
        for key, m in sorted(MATERIALS.items())
    }

    beams = {"thermal": nxs.thermal_beam(), "cold_poly_35K": nxs.cold_poly_beam()}
    samples = {"H2O": 1.0, "C2H4": 0.95, "Fe": 7.874, "Li": 0.534,
               "LiNi0.8Mn0.1Co0.1O2": 4.8}
    out["beams"] = {
        f"{formula}/{b}": float(mu_n_for_beam(parse_formula(formula), rho, beam))
        for formula, rho in samples.items()
        for b, beam in beams.items()
    }

    phantom = nxs.make_phantom("composite", N=32)
    labels = phantom.label_vol
    out["phantom"] = {m.symbol: int((labels == i).sum())
                      for i, m in enumerate(phantom.materials)}

    x, n = nxs.make_sinogram_pair(phantom, n_angles=24, use_astra=False, verbose=False)
    out["sinograms"] = {"xray": _stats(x["sino_lam"]), "neutron": _stats(n["sino_lam"])}

    cfg = nxs.ArtifactConfig(neutron_scatter=True, xray_scatter=True, detector_psf=True,
                             apply_bh_correction=True)
    xa, na = nxs.inject_sinogram_artifacts(x, n, cfg)
    out["detection_chain"] = {"xray": _stats(xa["sino_lam"]),
                              "neutron": _stats(na["sino_lam"])}

    rec = {}
    for name, sino in (("xray", x), ("neutron", n)):
        vol = nxs.reconstruct(sino, use_astra=False, remove_rings=False)
        rec[name] = {m.symbol: float(vol[labels == i].mean())
                     for i, m in enumerate(phantom.materials)}
    out["reconstruction"] = rec
    return out


def _flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, f"{prefix}{k}/")
        else:
            yield f"{prefix}{k}", v


@pytest.fixture(scope="module")
def values():
    return compute_values()


def test_update_reference(request, values):
    if not request.config.getoption("--update-reference"):
        pytest.skip("pass --update-reference to regenerate the stored values")
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps(values, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")


@pytest.mark.parametrize("group", sorted(RTOL))
def test_simulated_numbers_unchanged(request, values, group):
    if request.config.getoption("--update-reference"):
        pytest.skip("reference being regenerated")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))[group]
    new = dict(_flatten(values[group]))
    old = dict(_flatten(reference))
    assert new.keys() == old.keys(), (
        f"keys changed: added {sorted(new.keys() - old.keys())}, "
        f"removed {sorted(old.keys() - new.keys())}")
    bad = [f"{k}: {old[k]!r} -> {new[k]!r}" for k in sorted(old)
           if not np.isclose(new[k], old[k], rtol=RTOL[group], atol=1e-12)]
    assert not bad, (
        f"{len(bad)} {group} value(s) changed (rtol={RTOL[group]}). If intended, run "
        "`pytest tests/test_regression.py --update-reference` and add a CHANGELOG "
        "entry.\n  " + "\n  ".join(bad[:20]))
