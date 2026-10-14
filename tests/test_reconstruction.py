"""CPU reconstruction paths.  ASTRA / TomoPy are never required: the GPU-only
iterative solvers are exercised against a small dense NumPy stand-in for the
four ``astra`` calls they use, injected via ``monkeypatch``."""

import itertools
import sys
import types
import warnings

import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.acquisition.projector import line_integrals
from neutron_xray_sim.reconstruction import reconstructor as rc

VOXEL_CM = 0.05


def _grid(n):
    yy, xx = np.mgrid[:n, :n] - n // 2
    return yy, xx, np.hypot(yy, xx)


def _disc_sino(n=48, radius=14.0, mu=2.0, n_angles=60):
    """Exact parallel-beam sinogram of a uniform disc (chord length × μ)."""
    s = np.arange(n) - n // 2
    proj = 2 * mu * np.sqrt(np.clip(radius ** 2 - s ** 2, 0, None)) * VOXEL_CM
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sino = np.tile(proj, (n_angles, 1))[:, None, :].astype(np.float32)
    return {"sino_lam": sino, "angles_deg": angles, "voxel_cm": VOXEL_CM}


def _sino_of(img, n_angles=60):
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sino = line_integrals(img[None].astype(np.float32), angles, use_astra=False) * VOXEL_CM
    return {"sino_lam": sino, "angles_deg": angles, "voxel_cm": VOXEL_CM}


CPU = dict(use_astra=False, remove_rings=False)


# ── Algorithm names ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, canonical", [
    ("fbp", "FBP"), ("RAMP", "FBP"), ("backprojection", "FBP"),
    ("grid", "GRIDREC"), ("conjugate", "CGLS"), ("mlem", "EM"),
    ("ossart", "OSSART"), ("tv", "TV_MIN"), ("Total-Variation", "TV_MIN"),
    ("nesterov", "NESTEROV_SIRT"), ("sirt-nesterov", "NESTEROV_SIRT"),
])
def test_algorithm_aliases(name, canonical):
    assert rc._resolve_algorithm(name) == canonical


def test_every_algorithm_name_resolves_to_itself():
    for alg in nxs.AVAILABLE_ALGORITHMS:
        assert rc._resolve_algorithm(alg.lower()) == alg


@pytest.mark.xfail(strict=True, reason=(
    "_resolve_algorithm maps '-' → '_' before the alias lookup "
    "(reconstructor.py:143) but the hyphenated alias keys at :121/:128/:129 "
    "keep their '-', so they can never match"))
@pytest.mark.parametrize("name", ["ram-lak", "ml-em", "os-sart"])
def test_documented_hyphenated_aliases_resolve(name):
    rc._resolve_algorithm(name)


def test_unknown_algorithm_lists_options():
    with pytest.raises(ValueError, match="Available: FBP, GRIDREC"):
        rc._resolve_algorithm("magic")


# ── Filtered back-projection ─────────────────────────────────────────────────

@pytest.mark.parametrize("filt", ["ramp", "ram-lak", "shepp-logan", "cosine", "hann",
                                  "hamming"])
def test_fbp_recovers_disc_attenuation(filt):
    # Exact disc sinogram (μ = 2 cm⁻¹, R = 14 px): the interior must come
    # back in cm⁻¹ (divided by voxel_cm).  Measured bias is +0.7–0.8% for
    # every filter (pixel discretisation of the ramp); 2% leaves margin.
    _, _, r = _grid(48)
    vol = nxs.reconstruct(_disc_sino(), filter_name=filt, clip_negative=False, **CPU)
    assert vol.shape == (1, 48, 48) and vol.dtype == np.float32
    inner = vol[0][r < 11]
    assert inner.mean() == pytest.approx(2.0, rel=0.02)
    assert inner.std() < 0.02
    assert np.abs(vol[0][(r > 17) & (r < 23)]).max() < 0.06     # air stays ≈ 0


def test_fbp_apodising_filters_suppress_noise():
    # Noise-only sinogram: the window functions roll off the ramp's high
    # frequencies, so reconstructed noise must strictly decrease
    # ramp > shepp-logan > cosine > hamming > hann (verified over 5 seeds,
    # the smallest gap is 7%).
    _, _, r = _grid(48)
    noise = np.random.default_rng(0).normal(0, 1, (60, 1, 48)).astype(np.float32)
    d = {"sino_lam": noise, "angles_deg": np.linspace(0, 180, 60, endpoint=False)}
    sd = [nxs.reconstruct(d, filter_name=f, clip_negative=False, **CPU)[0][r < 18].std()
          for f in ("ramp", "shepp-logan", "cosine", "hamming", "hann")]
    assert all(a > b for a, b in zip(sd, sd[1:]))


def test_unknown_filter_falls_back_to_shepp_logan():
    d = _disc_sino()
    np.testing.assert_array_equal(nxs.reconstruct(d, filter_name="bogus", **CPU),
                                  nxs.reconstruct(d, filter_name="shepp-logan", **CPU))
    assert rc._astra_filter("SheppLogan") == "shepp-logan"
    assert rc._astra_filter("Ramp") == "ram-lak"


def test_unit_scaling_and_clipping():
    d = _disc_sino()
    raw = nxs.reconstruct({k: v for k, v in d.items() if k != "voxel_cm"},
                          clip_negative=False, **CPU)
    cm = nxs.reconstruct(d, clip_negative=False, **CPU)
    np.testing.assert_allclose(cm, raw / VOXEL_CM, rtol=1e-6)
    # FBP Gibbs undershoot outside the disc is negative until clipped.
    assert cm.min() < 0
    assert nxs.reconstruct(d, **CPU).min() == 0.0
    thr = nxs.reconstruct(d, clip_threshold=0.5, **CPU)
    _, _, r = _grid(48)
    assert np.all((thr == 0) | (thr >= 0.5))
    assert np.all(thr[0][r > 17] == 0) and np.all(thr[0][r < 11] > 0)


def test_center_offset_correction():
    # A sinogram whose rotation axis is shifted by +2 px is restored exactly
    # by center_offset = -2 (integer shift, linear interpolation is exact).
    from scipy.ndimage import shift
    _, _, r = _grid(48)
    img = ((r < 10) * 1.0 + (np.hypot(*(np.mgrid[:48, :48] - [[[30]], [[20]]])) < 4)).astype(
        np.float32)
    d = _sino_of(img)
    ref = nxs.reconstruct(d, **CPU)
    off = dict(d, sino_lam=shift(d["sino_lam"], (0, 0, 2.0), mode="nearest"))
    assert np.sqrt(np.mean((nxs.reconstruct(off, **CPU) - ref) ** 2)) > 0.2
    np.testing.assert_allclose(nxs.reconstruct(off, center_offset=-2.0, **CPU), ref,
                               atol=1e-5)


def test_reconstruct_pair_recovers_both_modalities(small_phantom, clean_sinos, capsys):
    x, n = clean_sinos
    vx, vn = nxs.reconstruct_pair(x, n, use_astra=False, remove_rings=False, verbose=False)
    assert capsys.readouterr().out == ""
    assert vx.shape == vn.shape == small_phantom.shape
    np.testing.assert_array_equal(vn, nxs.reconstruct(n, **CPU))
    # Each output tracks its own modality's μ map, not the other one
    # (measured r = 0.94 / 0.98 matched, < 0.2 crossed).
    s = small_phantom.shape[0] // 2
    mu_x = small_phantom.mu_x_vols[list(nxs.XRAY_E_KEV).index(80)]

    def corr(a, b):
        return np.corrcoef(a[s].ravel(), b[s].ravel())[0, 1]

    assert corr(vn, small_phantom.mu_n_vol) > 0.95
    assert corr(vx, mu_x) > 0.9
    assert corr(vx, small_phantom.mu_n_vol) < 0.5 and corr(vn, mu_x) < 0.5
    # The HDPE body (label 2, 6 318 voxels after 2× erosion) is recovered
    # to 1% of its tabulated μₙ = 2.18 cm⁻¹ (measured: −0.02%).
    st = nxs.material_stats(vn, small_phantom.label_vol, [2], erode=2)
    assert st[2][0] == pytest.approx(small_phantom.materials[2].mu_n, rel=0.01)
    nxs.reconstruct_pair(x, n, use_astra=False, verbose=True)
    out = capsys.readouterr().out
    assert "Reconstructing with FBP" in out and "Done" in out


# ── Ring removal ─────────────────────────────────────────────────────────────

def test_ring_removal_subtracts_column_stripe():
    # A detector column with a constant offset (vertical stripe) on a smooth
    # sinogram: its contrast against its neighbours drops from 0.3 to ~0.01
    # (measured 0.013), and columns outside the la = 11 median window (and
    # away from the detector edges) are untouched.
    rng = np.random.default_rng(0)
    base = (np.linspace(0.2, 1.4, 48)[None, :] + rng.normal(0, 0.05, (60, 1)))
    base = base.astype(np.float32)
    striped = base.copy()
    striped[:, 20] += 0.3
    out = rc._remove_rings_vo(striped)

    def contrast(s):
        return np.mean(s[:, 20] - 0.5 * (s[:, 19] + s[:, 21]))

    assert contrast(striped) == pytest.approx(0.3, abs=1e-5)
    assert abs(contrast(out)) < 0.03
    far = np.r_[6:14, 27:41]
    np.testing.assert_array_equal(out[:, far], base[:, far])


def test_ring_removal_suppresses_ring_in_reconstruction():
    # Stripe in an air column outside a centred disc → ring at r ≈ 21 px.
    _, _, r = _grid(48)
    d = _sino_of(((r < 10) * 2.0).astype(np.float32))
    striped = dict(d, sino_lam=d["sino_lam"].copy())
    striped["sino_lam"][:, :, 3] += 0.05
    ref = nxs.reconstruct(d, use_astra=False, remove_rings=False, clip_negative=False)
    annulus = (r > 18) & (r < 23)

    def ring_rms(remove):
        v = nxs.reconstruct(striped, use_astra=False, remove_rings=remove,
                            clip_negative=False)
        return np.sqrt(np.mean((v - ref)[0][annulus] ** 2))

    assert ring_rms(False) > 0.015                # measured 0.022
    assert ring_rms(True) < 0.1 * ring_rms(False)  # measured 0.0004


def test_ring_removal_noop_without_column_structure():
    flat = np.full((10, 16), 0.7, np.float32)
    out = rc._remove_rings_vo(flat)
    np.testing.assert_array_equal(out, flat)
    assert out is not flat


@pytest.mark.xfail(strict=True, reason=(
    "_remove_rings_vo (reconstructor.py:196-200) subtracts the median-filter "
    "residual of the column mean, which removes real object structure "
    "narrower than la=11 px; remove_rings=True is the default"))
def test_ring_removal_preserves_stripe_free_object():
    # Small (R = 6 px) disc, no stripes: the default ring removal should not
    # change the reconstructed attenuation.  Measured: 1.99 → 1.52 cm⁻¹ (−24%).
    _, _, r = _grid(48)
    d = _sino_of(((r < 6) * 2.0).astype(np.float32))
    v = nxs.reconstruct(d, use_astra=False, remove_rings=True)
    assert v[0][r < 3].mean() == pytest.approx(2.0, rel=0.05)


@pytest.mark.xfail(strict=True, reason=(
    "ring_snr / snr is accepted (reconstructor.py:178, :427) but never used"))
def test_ring_snr_changes_aggressiveness():
    rng = np.random.default_rng(0)
    s = rng.normal(1.0, 0.05, (60, 48)).astype(np.float32)
    s[:, 20] += 0.3
    assert not np.array_equal(rc._remove_rings_vo(s, snr=0.5),
                              rc._remove_rings_vo(s, snr=50.0))


# ── Fallbacks for missing optional back-ends ─────────────────────────────────

@pytest.fixture
def no_backends(monkeypatch):
    monkeypatch.setattr(rc, "_astra_ok", lambda: False)
    monkeypatch.setattr(rc, "_tomopy_ok", lambda: False)


@pytest.mark.parametrize("alg", ["SIRT", "SART", "CGLS", "EM", "OSSART", "TV_MIN",
                                 "NESTEROV_SIRT"])
def test_iterative_without_astra_falls_back_to_fbp(no_backends, alg):
    d = _disc_sino(n_angles=20)
    with pytest.warns(UserWarning, match=f"falling back to skimage FBP.*{alg}"):
        vol = nxs.reconstruct(d, algorithm=alg, use_astra=True)
    np.testing.assert_array_equal(vol, nxs.reconstruct(d, use_astra=False))


def test_gridrec_without_tomopy_falls_back_to_fbp(no_backends):
    d = _disc_sino(n_angles=20)
    with pytest.warns(UserWarning, match="TomoPy not available"):
        vol = nxs.reconstruct(d, algorithm="gridrec", use_astra=False)
    np.testing.assert_array_equal(vol, nxs.reconstruct(d, use_astra=False))


def test_fbp_without_astra_is_silent(no_backends):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nxs.reconstruct(_disc_sino(n_angles=20), algorithm="FBP", use_astra=True)


def test_backend_probes_report_missing_modules(monkeypatch):
    # A None entry in sys.modules makes `import x` raise ImportError.
    monkeypatch.setitem(sys.modules, "astra", None)
    monkeypatch.setitem(sys.modules, "tomopy", None)
    assert rc._astra_ok() is False and rc._tomopy_ok() is False


def test_gridrec_with_tomopy():
    pytest.importorskip("tomopy")
    _, _, r = _grid(48)
    vol = nxs.reconstruct(_disc_sino(), algorithm="gridrec", use_astra=False,
                          remove_rings=False)
    assert vol[0][r < 11].mean() == pytest.approx(2.0, rel=0.05)


# ── Iterative solvers against a NumPy stand-in for ASTRA ────────────────────

def _system_matrix(n, angles_rad):
    """Dense pixel-driven projector with ASTRA scaling (each pixel deposits
    unit weight per angle), matching the CPU projector's geometry."""
    c = np.arange(n) - n // 2
    X, Y = np.meshgrid(c, c)
    pix = np.arange(n * n)
    A = np.zeros((len(angles_rad), n, n * n))
    for i, th in enumerate(angles_rad):
        t = (X * np.cos(th) - Y * np.sin(th)).ravel() + n // 2
        j0 = np.floor(t).astype(int)
        f = t - j0
        for j, w in ((j0, 1 - f), (j0 + 1, f)):
            ok = (j >= 0) & (j < n)
            np.add.at(A[i], (j[ok], pix[ok]), w[ok])
    return A.reshape(-1, n * n)


class _FakeAstra(types.ModuleType):
    """Implements only the astra calls used by _nesterov_sirt / _tv_min_admm.

    Like real ASTRA, ``create_sino`` / ``create_backprojection`` allocate a
    data object whose id must be released with ``astra.data2d.delete``;
    ``live`` tracks the ones still allocated."""

    def __init__(self):
        super().__init__("astra")
        self.projectors = {}
        self.live = set()
        self._ids = itertools.count(1)
        self.projector = types.SimpleNamespace(delete=self.projectors.pop)
        self.data2d = types.SimpleNamespace(delete=self._delete)

    def _new_data(self, arr):
        i = next(self._ids)
        self.live.add(i)
        return i, arr.astype(np.float32)

    def _delete(self, ids):
        for i in np.atleast_1d(ids):
            self.live.discard(int(i))

    def create_vol_geom(self, rows, cols):
        return (rows, cols)

    def create_proj_geom(self, kind, width, n_det, angles):
        return (n_det, np.asarray(angles))

    def create_projector(self, kind, proj_geom, vol_geom):
        pid = next(self._ids)
        self.projectors[pid] = (_system_matrix(*proj_geom), proj_geom, vol_geom)
        return pid

    def create_sino(self, img, pid):
        A, (n, ang), _ = self.projectors[pid]
        return self._new_data((A @ np.ravel(img)).reshape(len(ang), n))

    def create_backprojection(self, sino, pid):
        A, _, vol_geom = self.projectors[pid]
        return self._new_data((A.T @ np.ravel(sino)).reshape(vol_geom))


@pytest.fixture
def fake_astra(monkeypatch):
    fake = _FakeAstra()
    monkeypatch.setitem(sys.modules, "astra", fake)
    return fake


def _iterative_problem(n=24, n_angles=30):
    yy, xx, r = _grid(n)
    img = (r < 8) * 0.1 + ((np.abs(yy - 2) < 2) & (np.abs(xx + 1) < 3)) * 0.2
    ang = np.radians(np.linspace(0, 180, n_angles, endpoint=False))
    A = _system_matrix(n, ang)
    p = (A @ img.ravel()).reshape(n_angles, n).astype(np.float32)

    def residual(x):
        return np.linalg.norm(A @ x.ravel() - p.ravel()) / np.linalg.norm(p)

    return img, p, ang, residual


def test_fake_astra_matches_cpu_projector():
    # Guard: the stand-in must model the same geometry as the real projector,
    # else the tests below would validate the solvers against the wrong A.
    yy, xx, r = _grid(32)
    img = ((r < 10) * 1.0 + ((np.abs(yy - 3) < 3) & (np.abs(xx + 2) < 4)) * 1.5)
    ang = np.linspace(0, 180, 40, endpoint=False)
    ref = line_integrals(img[None].astype(np.float32), ang, use_astra=False)[:, 0, :]
    fake = (_system_matrix(32, np.radians(ang)) @ img.ravel()).reshape(40, 32)
    assert np.corrcoef(ref.ravel(), fake.ravel())[0, 1] > 0.999


def test_nesterov_sirt_residual_decreases(fake_astra):
    img, p, ang, residual = _iterative_problem()
    res = {}
    for n_iter in (5, 20, 50, 100):
        x = rc._nesterov_sirt(p, ang, n_iter, 24)
        assert x.min() >= 0                           # non-negativity
        res[n_iter] = residual(x)
    assert res[5] > res[20] > res[50] > res[100]
    assert res[100] < 2e-3                            # measured 9.8e-4
    # Image error converges more slowly than the data residual (measured 0.059).
    assert np.linalg.norm(x - img) / np.linalg.norm(img) < 0.08
    assert fake_astra.projectors == {}                # projector released


def test_nesterov_beats_plain_landweber(fake_astra):
    # Momentum is the point of the method: at the same step size and
    # iteration count, plain projected gradient (Landweber/SIRT) lags far
    # behind (measured residual 0.058 vs 0.015 at 20 iterations).
    _, p, ang, residual = _iterative_problem()
    A = _system_matrix(24, ang)
    step = 1.0 / (A.sum(0).max() * A.sum(1).max())
    x = np.zeros(A.shape[1])
    for _ in range(20):
        x = np.maximum(0.0, x - step * A.T @ (A @ x - p.ravel()))
    assert residual(rc._nesterov_sirt(p, ang, 20, 24)) < 0.5 * residual(x)


def test_nesterov_sirt_via_reconstruct(fake_astra):
    # Full dispatch with use_astra=True: units scaled to cm⁻¹ like FBP.
    _, _, r = _grid(32)
    d = _sino_of(((r < 10) * 2.0).astype(np.float32), n_angles=40)
    vol = nxs.reconstruct(d, algorithm="nesterov", n_iter=100, use_astra=True,
                          remove_rings=False)
    assert vol[0][r < 7].mean() == pytest.approx(2.0, rel=0.02)


@pytest.mark.xfail(strict=True, reason=(
    "_tv_min_admm step = 1/(max(Aᵀ1)+ρ) (reconstructor.py:360) ignores the "
    "forward-projection row sum, so the gradient step is ~20× too large and "
    "x oscillates between an overshoot and 0 instead of converging"))
def test_tv_min_admm_residual_decreases(fake_astra):
    img, p, ang, residual = _iterative_problem()
    res = [residual(rc._tv_min_admm(p, ang, n, 24)) for n in (5, 20, 50)]
    assert res[0] > res[1] > res[2]
    assert res[2] < 0.1


def test_tv_min_admm_contract(fake_astra):
    # Independent of convergence: output shape, non-negativity, cleanup.
    _, p, ang, _ = _iterative_problem()
    x = rc._tv_min_admm(p, ang, 3, 24, lambda_tv=0.05)
    assert x.shape == (24, 24) and x.min() >= 0
    assert fake_astra.projectors == {}


@pytest.mark.xfail(strict=True, reason=(
    "the ids returned by astra.create_sino / create_backprojection are "
    "discarded and never passed to astra.data2d.delete (reconstructor.py:299-313, "
    ":359-380): 2 ASTRA data objects leak per iteration per slice"))
@pytest.mark.parametrize("solver", [rc._nesterov_sirt, rc._tv_min_admm],
                         ids=["nesterov_sirt", "tv_min_admm"])
def test_iterative_solvers_release_astra_data(fake_astra, solver):
    _, p, ang, _ = _iterative_problem()
    solver(p, ang, 5, 24)
    assert fake_astra.live == set()


def test_fbp_cuda_failure_falls_back_to_skimage(monkeypatch):
    # ASTRA importable but the GPU call fails (no device / driver error):
    # warn and fall back to the CPU iradon for that slice.
    def broken(*args, **kwargs):
        raise RuntimeError("no CUDA device")

    fake = types.ModuleType("astra")
    fake.create_vol_geom = fake.create_proj_geom = lambda *args: None
    fake.data2d = types.SimpleNamespace(create=broken, delete=broken)
    fake.algorithm = types.SimpleNamespace(delete=broken)
    monkeypatch.setitem(sys.modules, "astra", fake)
    monkeypatch.setattr(rc, "_astra_ok", lambda: True)
    d = _disc_sino(n_angles=20)
    with pytest.warns(UserWarning, match=r"FBP_CUDA failed \(no CUDA device\)"):
        vol = nxs.reconstruct(d, use_astra=True)
    monkeypatch.setattr(rc, "_astra_ok", lambda: False)
    np.testing.assert_array_equal(vol, nxs.reconstruct(d, use_astra=False))
