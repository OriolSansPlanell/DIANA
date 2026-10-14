import os
import subprocess
import sys

import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.acquisition.noise import DOSE_PRESETS, DoseConfig, rng_for


# ── Configuration ────────────────────────────────────────────────────────────

def test_dose_config_validation_and_label():
    with pytest.raises(ValueError, match="I0_neutron"):
        DoseConfig(I0_neutron=0)
    with pytest.raises(ValueError, match="flat_counts_xray"):
        DoseConfig(flat_counts_xray=-1.0)
    cfg = DoseConfig(1e5, 1e3)
    assert cfg.label == "Ix1e+05_In1e+03"
    assert not cfg.noiseless
    assert cfg.describe() == ("X-ray: I0=1e+05, ideal flat | "
                              "neutron: I0=1e+03, ideal flat")
    half = DoseConfig(np.inf, 1e3, flat_counts_neutron=1e4)
    assert "X-ray: noiseless" in half.describe() and "flat=1e+04" in half.describe()
    assert DOSE_PRESETS["noiseless"].noiseless


def test_dose_presets_neutron_flux_is_tenfold_lower():
    # Documented flux ratio between a lab tube and a cold-neutron beamline.
    for name, cfg in DOSE_PRESETS.items():
        if not cfg.noiseless:
            assert cfg.I0_xray == pytest.approx(10 * cfg.I0_neutron), name


# ── Reproducible random streams ──────────────────────────────────────────────

def test_rng_for_is_reproducible_and_tag_specific():
    a = rng_for(42, "xray", 1e4).standard_normal(1000)
    np.testing.assert_array_equal(a, rng_for(42, "xray", 1e4).standard_normal(1000))
    for other in (rng_for(42, "neutron", 1e4), rng_for(43, "xray", 1e4),
                  rng_for(42, "xray", 1e3)):
        assert not np.array_equal(a, other.standard_normal(1000))
    # Different modalities are statistically uncorrelated (|r| < 4/sqrt(n)).
    n = 100_000
    r = np.corrcoef(rng_for(42, "xray", 1e4).standard_normal(n),
                    rng_for(42, "neutron", 1e4).standard_normal(n))[0, 1]
    assert abs(r) < 4 / np.sqrt(n)


def test_rng_for_is_stable_across_processes():
    # Must not depend on Python's salted hash(): a study is re-runnable later.
    code = ("from neutron_xray_sim.acquisition.noise import rng_for;"
            "print(rng_for(42, 'xray', 1e4).integers(0, 2**62))")
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True)
    assert int(out.stdout) == rng_for(42, "xray", 1e4).integers(0, 2**62)


# ── Poisson sampler ──────────────────────────────────────────────────────────

def test_poisson_counts_have_variance_equal_to_mean():
    # Invert the noisy optical depth back to detected counts: they must be
    # integers with mean I0·exp(-λ) and Fano factor (var/mean) = 1.
    I0, lam = 1e3, 1.0
    out = nxs.add_poisson_noise(np.full(200_000, lam), I0, rng=np.random.default_rng(0),
                                dtype=np.float64)
    counts = I0 * np.exp(-out)
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-6)
    mean = I0 * np.exp(-lam)
    # 5-sigma tolerances: SEM = sqrt(mean/n) ≈ 0.04; sd(var/mean) ≈ sqrt(2/n) ≈ 0.003.
    assert counts.mean() == pytest.approx(mean, abs=5 * np.sqrt(mean / counts.size))
    assert counts.var() / counts.mean() == pytest.approx(1.0, abs=0.016)


def test_noise_falls_as_inverse_sqrt_dose():
    rng = np.random.default_rng(1)
    lam = np.full(100_000, 1.0)
    sd = [nxs.add_poisson_noise(lam, I0, rng=rng).std() for I0 in (1e3, 1e4, 1e5)]
    assert sd[0] > sd[1] > sd[2]
    assert sd[0] / sd[1] == pytest.approx(np.sqrt(10), rel=0.03)
    assert sd[1] / sd[2] == pytest.approx(np.sqrt(10), rel=0.03)


def test_log_estimator_bias_matches_second_order_theory():
    # E[-ln(N/I0)] ≈ λ + Var/2 = λ + exp(λ)/(2·I0): the log transform biases
    # optical depth high.  SEM of the mean here is ~5% of the expected bias.
    I0, lam = 1e3, 2.0
    out = nxs.add_poisson_noise(np.full(200_000, lam), I0, rng=np.random.default_rng(2),
                                dtype=np.float64)
    assert out.mean() - lam == pytest.approx(np.exp(lam) / (2 * I0), rel=0.2)


def test_zero_counts_floored_at_half_a_count():
    out = nxs.add_poisson_noise(np.full(1000, 50.0), 100.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out, np.log(100.0 / 0.5), rtol=1e-6)


def test_add_poisson_noise_noiseless_and_dtype(rng):
    lam = rng.uniform(0, 2, (3, 4, 5))
    out = nxs.add_poisson_noise(lam, np.inf)
    assert out.dtype == np.float32 and out is not lam
    np.testing.assert_allclose(out, lam, rtol=1e-6)
    a = nxs.add_poisson_noise(lam, 1e3, rng=rng_for(0, "n"))
    b = nxs.add_poisson_noise(lam, 1e3, rng=rng_for(0, "n"))
    np.testing.assert_array_equal(a, b)
    assert nxs.add_poisson_noise(lam, 1e3).shape == lam.shape        # default rng


@pytest.mark.parametrize("flat", [None, 5e4])
def test_measured_sigma_matches_prediction(flat):
    # First-order error propagation is accurate to O(exp(λ)/I0) ≈ 0.2% at
    # I0 = 1e4, λ ≤ 3; each bin holds ~33k samples, so the sampling error on
    # sigma is ~0.4%.  A 3% band is therefore > 5 sigma.
    I0 = 1e4
    rng = np.random.default_rng(4)
    lam = rng.uniform(0, 3, 400_000)
    noisy = nxs.add_poisson_noise(lam, I0, rng=rng, flat_counts=flat)
    m = nxs.measure_sigma_lambda(lam, noisy, n_bins=12, lam_range=(0, 3))
    pred = nxs.predicted_sigma_lambda(m["lam_centres"], I0, flat_counts=flat)
    np.testing.assert_allclose(m["sigma"], pred, rtol=0.03)
    assert m["count"].sum() == lam.size


def test_measured_sigma_on_phantom_sinogram(small_phantom):
    # Same check with the default λ-range on a real neutron sinogram.
    angles = np.linspace(0, 180, 60, endpoint=False)
    lam = nxs.project_neutron(small_phantom, angles, use_astra=False)["sino_lam"]
    noisy = nxs.add_poisson_noise(lam, 1e3, rng=rng_for(0, "neutron", 1e3))
    m = nxs.measure_sigma_lambda(lam, noisy)
    ok = m["count"] > 1000                      # sd(sigma)/sigma < 1/sqrt(2n) ≈ 2.2%
    ratio = m["sigma"][ok] / nxs.predicted_sigma_lambda(m["lam_centres"][ok], 1e3)
    np.testing.assert_allclose(ratio, 1.0, atol=0.1)
    assert m["lam_centres"][-1] < np.percentile(lam, 99.5)


def test_flat_field_noise_adds_in_quadrature():
    lam = np.zeros(200_000)
    rng = np.random.default_rng(5)
    ideal = nxs.add_poisson_noise(lam, 1e3, rng=rng).std()
    flat = nxs.add_poisson_noise(lam, 1e3, rng=rng, flat_counts=1e3).std()
    assert flat / ideal == pytest.approx(np.sqrt(2), rel=0.02)
    assert nxs.predicted_sigma_lambda(0.0, 1e3, flat_counts=1e3) == pytest.approx(
        np.sqrt(2e-3))
    np.testing.assert_array_equal(nxs.predicted_sigma_lambda([0.0, 2.0], np.inf), 0.0)


def test_measure_sigma_sparse_bins_are_nan():
    lam = np.r_[np.zeros(1000), np.full(5, 1.0), np.full(1000, 2.0)]
    noisy = nxs.add_poisson_noise(lam, 1e4, rng=np.random.default_rng(0))
    m = nxs.measure_sigma_lambda(lam, noisy, n_bins=4, lam_range=(0, 2))
    np.testing.assert_array_equal(m["count"], [1000, 0, 5, 1000])
    assert np.isnan(m["sigma"][1]) and np.isnan(m["sigma"][2])     # ≤ 8 samples
    assert np.all(np.isfinite(m["sigma"][[0, 3]]))


@pytest.mark.xfail(strict=True, reason=(
    "measure_sigma_lambda clips out-of-range λ into the edge bins "
    "(noise.py:319) instead of discarding them"))
def test_measure_sigma_ignores_samples_outside_range():
    I0 = 1e4
    rng = np.random.default_rng(6)
    lam = rng.uniform(0, 3, 400_000)
    noisy = nxs.add_poisson_noise(lam, I0, rng=rng)
    m = nxs.measure_sigma_lambda(lam, noisy, n_bins=6, lam_range=(0, 1.5))
    pred = nxs.predicted_sigma_lambda(m["lam_centres"], I0)
    np.testing.assert_allclose(m["sigma"], pred, rtol=0.03)


# ── Streaming (memmap) version ───────────────────────────────────────────────

def test_streaming_matches_in_memory(rng):
    # Chunking along the leading axis draws in the same order as one call.
    lam = rng.uniform(0, 2, (10, 7, 9)).astype(np.float32)
    ref = nxs.add_poisson_noise(lam, 1e3, rng=np.random.default_rng(5))
    out = nxs.add_poisson_noise_streaming(lam, np.empty_like(lam), 1e3,
                                          rng=np.random.default_rng(5),
                                          slice_axis=0, chunk=3)
    np.testing.assert_array_equal(out, ref)


def test_streaming_in_place_on_memmap(tmp_path, capsys):
    lam = np.full((20, 40, 30), 1.0, dtype=np.float32)
    mm = np.lib.format.open_memmap(tmp_path / "sino.npy", mode="w+",
                                   dtype=np.float32, shape=lam.shape)
    mm[...] = lam
    out = nxs.add_poisson_noise_streaming(mm, mm, 1e4, rng=np.random.default_rng(0),
                                          chunk=16, progress=True)
    assert out is mm
    assert "40/40  done" in capsys.readouterr().out
    on_disk = np.load(tmp_path / "sino.npy")
    # Every chunk (incl. the short last one) is noisy with the right sigma.
    for sl in (slice(0, 16), slice(32, 40)):
        assert on_disk[:, sl].std() == pytest.approx(
            float(nxs.predicted_sigma_lambda(1.0, 1e4)), rel=0.05)


def test_streaming_noiseless_and_shape_check(rng):
    lam = rng.random((4, 3, 5)).astype(np.float32)
    dst = np.zeros_like(lam)
    assert nxs.add_poisson_noise_streaming(lam, dst, np.inf) is dst
    np.testing.assert_array_equal(dst, lam)
    with pytest.raises(ValueError, match="shape mismatch"):
        nxs.add_poisson_noise_streaming(lam, np.zeros((4, 3, 6)), 1e3)
    out = nxs.add_poisson_noise_streaming(lam, np.empty_like(lam), 1e3)   # default rng
    assert not np.array_equal(out, lam)


# ── Separability metrics ─────────────────────────────────────────────────────

def _two_phase(rng, n=20_000):
    labels = np.repeat([1, 2], n)
    return labels, rng.normal(0, 1, 2 * n)


def test_material_stats_and_erosion(rng):
    # A blurred square prism: erosion strips the partial-volume rim, so the
    # phase sigma drops to the pure noise level and the mean to the true value.
    # (Erosion is 3-D, so it also trims two slices off each z end.)
    from scipy.ndimage import gaussian_filter
    lab = np.zeros((8, 20, 20), int)
    lab[:, 5:15, 5:15] = 1
    vol = gaussian_filter(lab * 2.0, sigma=(0, 1, 1)) + rng.normal(0, 0.05, lab.shape)
    raw = nxs.material_stats(vol, lab, [1, 7])
    ero = nxs.material_stats(vol, lab, [1], erode=2)
    assert raw[1][2] == 800 and ero[1][2] == 4 * 6 * 6
    assert raw[1][1] > 0.25 and ero[1][1] < 0.07
    assert ero[1][0] == pytest.approx(2.0, abs=0.03)
    assert np.isnan(raw[7][0]) and raw[7][2] == 0


def test_cnr_and_d_prime(rng):
    labels, noise = _two_phase(rng)
    vol = np.where(labels == 1, 0.0, 2.0) + noise
    c = nxs.cnr(vol, labels, 1, 2)
    assert c == pytest.approx(2.0, rel=0.03)
    assert nxs.d_prime(vol, labels, 1, 2) == c
    assert np.isnan(nxs.cnr(vol, labels, 1, 99))
    assert nxs.cnr(np.where(labels == 1, 0.0, 1.0), labels, 1, 2) == np.inf


def test_joint_d_prime_geometry(rng):
    labels, ex = _two_phase(rng)
    _, en = _two_phase(rng)
    step = np.where(labels == 1, 0.0, 2.0)
    # Neutrons blind to the phase difference → nothing gained.
    r = nxs.joint_d_prime(step + ex, en, labels, 1, 2)
    assert r["gain"] == pytest.approx(1.0, abs=0.01) and r["angle_deg"] < 2
    # Independent noise, equal offsets → quadrature sum, 45°.
    r = nxs.joint_d_prime(step + ex, step + en, labels, 1, 2)
    assert r["gain"] == pytest.approx(np.sqrt(2), rel=0.03)
    assert r["angle_deg"] == pytest.approx(45, abs=2)
    assert r["d_joint"] >= max(r["d_x"], r["d_n"])


def test_joint_d_prime_correlated_noise(rng):
    # Shared noise (ρ = 0.9).  Offsets parallel to the noise axis are
    # redundant: d_joint = d·sqrt(2/(1+ρ)).  Anti-parallel offsets cut across
    # it: d_joint = d·sqrt(2/(1-ρ)), far beyond either modality alone.
    rho = 0.9
    labels, e = _two_phase(rng)
    _, e2 = _two_phase(rng)
    en = rho * e + np.sqrt(1 - rho ** 2) * e2
    step = np.where(labels == 1, 0.0, 2.0)
    par = nxs.joint_d_prime(step + e, step + en, labels, 1, 2)
    anti = nxs.joint_d_prime(step + e, -step + en, labels, 1, 2)
    assert par["d_joint"] == pytest.approx(2 * np.sqrt(2 / (1 + rho)), rel=0.03)
    assert anti["d_joint"] == pytest.approx(2 * np.sqrt(2 / (1 - rho)), rel=0.03)
    assert anti["gain"] > 4


@pytest.mark.xfail(strict=True, reason=(
    "angle_deg uses a Cholesky whitening (noise.py:481) that is not symmetric "
    "in the two modalities; with correlated noise it reports arcsin(d_n/d_joint)"))
def test_joint_d_prime_angle_is_symmetric(rng):
    # X-ray and neutron data are exchangeable here (same offset, same noise,
    # symmetric correlation), so by symmetry the separation is 45° and
    # swapping the modalities must give 90° − angle.
    rho = 0.9
    labels, e = _two_phase(rng)
    _, e2 = _two_phase(rng)
    step = np.where(labels == 1, 0.0, 2.0)
    vx, vn = step + e, step + rho * e + np.sqrt(1 - rho ** 2) * e2
    a = nxs.joint_d_prime(vx, vn, labels, 1, 2)["angle_deg"]
    b = nxs.joint_d_prime(vn, vx, labels, 1, 2)["angle_deg"]
    assert a + b == pytest.approx(90, abs=1)
    assert a == pytest.approx(45, abs=3)


def test_joint_d_prime_subsampling_and_missing(rng):
    labels, ex = _two_phase(rng)
    _, en = _two_phase(rng)
    step = np.where(labels == 1, 0.0, 2.0)
    full = nxs.joint_d_prime(step + ex, step + en, labels, 1, 2)
    kw = dict(max_voxels=2000, seed=3)
    sub = nxs.joint_d_prime(step + ex, step + en, labels, 1, 2, **kw)
    assert sub == nxs.joint_d_prime(step + ex, step + en, labels, 1, 2, **kw)
    # 2000 samples per phase: sd(d) ≈ sqrt(2/n + d²/4n) ≈ 1.6% of d → 8% is ~5 sigma.
    assert sub["d_joint"] == pytest.approx(full["d_joint"], rel=0.08)
    missing = nxs.joint_d_prime(ex, en, labels, 1, 99)
    assert all(np.isnan(v) for v in missing.values())


# ── Dose threshold ───────────────────────────────────────────────────────────

def test_rose_threshold_interpolates_in_log_dose():
    # Interpolation is piecewise log-linear; on a fine sweep of CNR = 0.1·sqrt(I0)
    # it recovers the exact crossing CNR = 5 at I0 = 2500.
    I0 = np.logspace(1, 5, 41)
    assert nxs.rose_dose_threshold(I0, 0.1 * np.sqrt(I0)) == pytest.approx(2500, rel=1e-3)
    # Exact log-linear interpolation, independent of input order.
    assert nxs.rose_dose_threshold([1e4, 1e2, 1e3], [9, 1, 3]) == pytest.approx(
        10 ** (3 + 1 / 3))
    # NaN points are skipped.
    assert nxs.rose_dose_threshold([1e2, 1e3, 1e4], [1, np.nan, 9]) == pytest.approx(1e3)


def test_rose_threshold_edge_cases():
    assert np.isnan(nxs.rose_dose_threshold([1e2, 1e3], [1, 2]))      # never reached
    assert nxs.rose_dose_threshold([1e2, 1e3], [6, 9]) == 1e2         # always above
    assert np.isnan(nxs.rose_dose_threshold([], []))


def test_dose_sweep_reconstruction():
    # End to end: Poisson noise → FBP.  FBP is linear, so the noise in the
    # volume scales as 1/sqrt(I0); CNR between water and iron rises with
    # dose and crosses the Rose criterion between 1e3 and 1e4 neutrons/px.
    b = nxs.PhantomBuilder(Nx=40, Ny=40, Nz=4, voxel_cm=0.05)
    b.add_cylinder("water", radius_cm=0.8, axis="z")
    b.add_cylinder("iron", radius_cm=0.3, axis="z", center_cm=(0, 0.3, 0.0))
    ph = b.build("dose")
    _, n = nxs.make_sinogram_pair(ph, n_angles=60, use_astra=False, verbose=False)
    kw = dict(remove_rings=False, clip_negative=False, use_astra=False)
    clean = nxs.reconstruct(n, **kw)
    doses = (1e2, 1e3, 1e4)
    sig, cnrs = [], []
    for I0 in doses:
        noisy = dict(n, sino_lam=nxs.add_poisson_noise(n["sino_lam"], I0,
                                                       rng=rng_for(0, "neutron", I0)))
        vol = nxs.reconstruct(noisy, **kw)
        sig.append(np.std(vol - clean))
        cnrs.append(nxs.cnr(vol, ph.label_vol, 1, 2, erode=1))
    assert sig[1] / sig[2] == pytest.approx(np.sqrt(10), rel=0.1)
    assert cnrs[0] < cnrs[1] < 5 < cnrs[2]
    assert 1e3 < nxs.rose_dose_threshold(doses, cnrs) < 1e4
