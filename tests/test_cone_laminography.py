"""Cone-beam / laminography geometry and the 3-D noise model (no GPU needed).

The ASTRA calls in ``acquisition.laminography`` need CUDA, so CI only checks that
they fail with a clear message; the geometry vectors and the noise model are
pure NumPy and are tested properly.
"""
import sys

import numpy as np
import pytest

from neutron_xray_sim.acquisition import laminography as lam
from neutron_xray_sim.acquisition.cone3d_geometry import (
    ConeGeometryConfig,
    _translation_positions_xy,
    make_cone_vec,
)


def _split(vectors):
    """ASTRA cone_vec rows are (source, detector centre, u, v), 3 values each."""
    return vectors[:, 0:3], vectors[:, 3:6], vectors[:, 6:9], vectors[:, 9:12]


def test_magnification_and_effective_pixel():
    cfg = ConeGeometryConfig(detector_pixel_um=150.0, sod_um=2400.0, sdd_um=120000.0)
    assert cfg.magnification == pytest.approx(50.0)
    assert cfg.effective_detector_pixel_um == pytest.approx(3.0)


def test_tomography_vectors_are_a_rigid_circular_orbit():
    cfg = ConeGeometryConfig(n_views=36, sod_um=1000.0, sdd_um=5000.0,
                             detector_pixel_um=10.0, rotation_range_deg=360.0)
    vectors, translations = make_cone_vec(cfg)
    src, det, u, v = _split(vectors.astype(float))
    assert vectors.shape == (36, 12)
    np.testing.assert_array_equal(translations, 0.0)
    # Source on a circle of radius SOD, detector at SDD from the source.
    np.testing.assert_allclose(np.linalg.norm(src, axis=1), 1000.0, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(det - src, axis=1), 5000.0, rtol=1e-6)
    # Pixel vectors have the pixel length; u is perpendicular to the beam axis;
    # v is the rotation axis (ASTRA z).
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 10.0, rtol=1e-6)
    np.testing.assert_allclose(np.einsum("ij,ij->i", u, det - src), 0.0, atol=1e-2)
    np.testing.assert_allclose(v, np.tile([0.0, 0.0, 10.0], (36, 1)), atol=1e-6)
    # Views are evenly spaced over the requested range (10° steps here).
    ang = np.unwrap(np.arctan2(det[:, 1], det[:, 0]))
    np.testing.assert_allclose(np.diff(np.rad2deg(ang)), 10.0, atol=1e-4)


def test_translated_laminography_moves_source_and_detector_together():
    cfg = ConeGeometryConfig(n_views=50, trajectory="translated_laminography",
                             translation_pattern="fermat", translation_radius_um=150.0,
                             sod_um=2000.0, sdd_um=8000.0, detector_pixel_um=20.0)
    vectors, translations = make_cone_vec(cfg)
    src, det, u, v = _split(vectors.astype(float))
    # The beam axis stays along +z with the detector parallel to the sample plane.
    np.testing.assert_allclose(det - src, np.tile([0, 0, 8000.0], (50, 1)), atol=1e-3)
    np.testing.assert_allclose(u[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(v[:, 2], 0.0, atol=1e-6)
    # Source and detector are shifted by the negative sample translation.
    np.testing.assert_allclose(src[:, :2], -translations[:, :2], atol=1e-3)
    np.testing.assert_allclose(src[:, 2], -2000.0, atol=1e-3)


@pytest.mark.parametrize("pattern", ["fermat", "circular", "raster"])
def test_translation_patterns_stay_inside_the_scan_area(pattern):
    cfg = ConeGeometryConfig(n_views=40, translation_pattern=pattern,
                             translation_radius_um=100.0)
    pts = _translation_positions_xy(cfg)
    assert pts.shape == (40, 3)
    np.testing.assert_array_equal(pts[:, 2], 0.0)
    if pattern == "raster":
        assert np.abs(pts[:, :2]).max() <= 100.0 + 1e-9
    else:
        assert np.hypot(pts[:, 0], pts[:, 1]).max() <= 100.0 + 1e-9
    if pattern == "circular":
        np.testing.assert_allclose(np.hypot(pts[:, 0], pts[:, 1]), 100.0)


def test_fermat_spiral_is_area_uniform():
    # r ∝ sqrt(i): equal numbers of points per unit area, so half the points lie
    # inside r = R / sqrt(2).
    cfg = ConeGeometryConfig(n_views=400, translation_pattern="fermat",
                             translation_radius_um=100.0)
    r = np.hypot(*_translation_positions_xy(cfg)[:, :2].T)
    assert np.mean(r < 100.0 / np.sqrt(2)) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("kwargs, match", [
    (dict(sod_um=5000.0, sdd_um=5000.0), "SOD < SDD"),
    (dict(sod_um=0.0), "SOD < SDD"),
    (dict(detector_shape_vu=(0, 10)), "positive"),
    (dict(detector_pixel_um=-1.0), "positive"),
    (dict(trajectory="helical"), "Unsupported trajectory"),
    (dict(trajectory="translated_laminography", translation_pattern="spiral"),
     "Unsupported translation pattern"),
])
def test_invalid_geometry_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_cone_vec(ConeGeometryConfig(n_views=4, **kwargs))


def test_gpu_functions_explain_that_astra_is_needed(monkeypatch):
    monkeypatch.setitem(sys.modules, "astra", None)   # force ImportError
    with pytest.raises(RuntimeError, match="ASTRA"):
        lam.forward_project_cone3d(np.zeros((4, 4, 4)), ConeGeometryConfig(n_views=2),
                                   (1.0, 1.0, 1.0))
    with pytest.raises(RuntimeError, match="ASTRA"):
        lam.reconstruct_cone3d(np.zeros((4, 2, 4)), ConeGeometryConfig(n_views=2),
                               (4, 4, 4), (1.0, 1.0, 1.0))


# ── line_integrals_to_noisy_log ──────────────────────────────────────────────

def test_noisy_log_is_unbiased_at_high_dose():
    p = np.linspace(0.0, 3.0, 4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    out = lam.line_integrals_to_noisy_log(p, incident_photons=1e12, detector_sigma_px=0)
    assert out.shape == p.shape and out.dtype == np.float32
    # Poisson σ(λ) = 1/sqrt(N0·T) ≤ 2e-5 here.
    np.testing.assert_allclose(out, p, atol=1e-4)


def test_noisy_log_noise_level_matches_poisson():
    p = np.full((1, 200, 200), 1.0, dtype=np.float32)
    n0 = 1e4
    out = lam.line_integrals_to_noisy_log(p, incident_photons=n0, detector_sigma_px=0)
    expected_sigma = 1.0 / np.sqrt(n0 * np.exp(-1.0))
    assert out.std() == pytest.approx(expected_sigma, rel=0.03)
    assert out.mean() == pytest.approx(1.0, abs=5 * expected_sigma / 200)


def test_noisy_log_is_seeded():
    p = np.ones((3, 8, 8), dtype=np.float32)
    a = lam.line_integrals_to_noisy_log(p, incident_photons=100, seed=1)
    b = lam.line_integrals_to_noisy_log(p, incident_photons=100, seed=1)
    c = lam.line_integrals_to_noisy_log(p, incident_photons=100, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_detector_blur_does_not_mix_views():
    # Axis 1 is the view axis in ASTRA (v, view, u) order: blur must not leak
    # between projections.
    p = np.zeros((9, 3, 9), dtype=np.float32)
    p[4, 1, 4] = 2.0
    out = lam.line_integrals_to_noisy_log(p, incident_photons=1e12, detector_sigma_px=1.0)
    np.testing.assert_allclose(out[:, [0, 2], :], 0.0, atol=1e-4)
    assert 0 < out[4, 1, 4] < 2.0          # blurred within the projection
    assert out[4, 1, 5] > 1e-3


def test_noisy_log_survives_opaque_rays():
    out = lam.line_integrals_to_noisy_log(np.full((2, 2, 2), 1e3, np.float32),
                                          incident_photons=1e3, detector_sigma_px=0)
    assert np.isfinite(out).all()
    assert out.max() <= np.log(1e3) + 1e-5   # counts are floored at 1


def test_noisy_log_rejects_non_positive_flux():
    with pytest.raises(ValueError, match="positive"):
        lam.line_integrals_to_noisy_log(np.zeros((1, 1, 1)), incident_photons=0)
