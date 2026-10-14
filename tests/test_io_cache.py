"""SimCache persistence: per-stage round trips, inventory and stale-cache checks.

Complements ``test_io.py`` (phantom round trip + simulation-driven caching).
"""

import json
import os

import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.io import SimCache, tag_to_slug


@pytest.fixture
def tiny_phantom():
    b = nxs.PhantomBuilder(N=8, voxel_cm=0.1)
    b.add_cylinder("aluminum", radius_cm=0.3)
    b.add_sphere("iron", radius_cm=0.15)
    return b.build("tiny")


def _result(tag, phantom, rng, cfg=None, n_angles=5):
    """A SimulationResult with random float64 data (no projection needed)."""
    shape = phantom.shape
    vx = phantom.mu_x_vols[nxs.DEFAULT_GT_ENERGY_IDX] + 0.01 * rng.standard_normal(shape)
    vn = phantom.mu_n_vol + 0.01 * rng.standard_normal(shape)
    sino = (n_angles, shape[0], shape[2])
    return nxs.SimulationResult(
        tag=tag, cfg=cfg or nxs.ArtifactConfig.clean(), phantom=phantom,
        vol_xray=vx, vol_neutron=vn,
        xray_sino={"sino_lam": rng.random(sino)},
        neutron_sino={"sino_lam": rng.random(sino)},
        histogram=nxs.compute_bimodal_histogram(vx, vn, bins=16), elapsed_s=1.25)


def _f32(a):
    # Every array is stored as float32 (documented); compare against that.
    return np.asarray(a).astype(np.float32)


# ── Runs ─────────────────────────────────────────────────────────────────────

def test_save_run_round_trip(tmp_path, tiny_phantom, rng):
    cfg = nxs.ArtifactConfig.misalignment_only()
    r = _result("Misalignment (3 vx)", tiny_phantom, rng, cfg=cfg)
    cache = SimCache(tmp_path)
    cache.save_run(r)

    for modality, vol in (("xray", r.vol_xray), ("neutron", r.vol_neutron)):
        out = cache.load_run_volume(r.tag, modality)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, _f32(vol))
    np.testing.assert_array_equal(cache.load_run_sinogram(r.tag, "xray"),
                                  _f32(r.xray_sino["sino_lam"]))
    np.testing.assert_array_equal(cache.load_run_sinogram(r.tag, "neutron"),
                                  _f32(r.neutron_sino["sino_lam"]))

    meta = cache.load_run_meta(r.tag)
    assert (meta["tag"], meta["slug"], meta["elapsed_s"]) == (
        r.tag, "misalignment_3_vx", 1.25)
    assert meta["schema_version"] == "1.0"
    # The stored artifact config fully reconstructs the one that was used
    # (JSON turns tuples into lists).
    stored = {k: tuple(v) if isinstance(v, list) else v
              for k, v in meta["artifact_cfg"].items()}
    assert nxs.ArtifactConfig(**stored) == cfg


def test_run_histogram_round_trip(tmp_path, tiny_phantom, rng):
    r = _result("h", tiny_phantom, rng)
    cache = SimCache(tmp_path)
    cache.save_run(r)
    h = cache.load_run_histogram("h")
    np.testing.assert_array_equal(h.H, r.histogram.H)  # counts are exact in float32
    np.testing.assert_allclose(h.x_edges, r.histogram.x_edges, rtol=1e-6)
    np.testing.assert_allclose(h.n_centres, r.histogram.n_centres, rtol=1e-6)
    assert h.extent == pytest.approx(r.histogram.extent, rel=1e-6)
    # Per-voxel samples are not stored, so only the binned voxels can be counted …
    assert h.total_voxels == int(r.histogram.H.sum())
    assert h.vol_x_flat.size == 0 and h.mask_flat is None
    # … and a GMM cannot be fitted to a re-loaded histogram.
    with pytest.raises(ValueError, match="recomputed from the volumes"):
        nxs.fit_gmm(h, n_components=2)


def test_save_run_histogram_replaces_the_histogram_only(tmp_path, tiny_phantom, rng):
    # Documented workflow: recompute the histogram from cached volumes, re-save it.
    r = _result("h", tiny_phantom, rng)
    cache = SimCache(tmp_path)
    cache.save_run(r)
    vx, vn = cache.load_run_volume("h", "xray"), cache.load_run_volume("h", "neutron")
    finer = nxs.compute_bimodal_histogram(vx, vn, bins=32)
    cache.save_run_histogram("h", finer)
    assert cache.load_run_histogram("h").H.shape == (32, 32)
    np.testing.assert_array_equal(cache.load_run_volume("h", "xray"), vx)


def test_single_array_savers_and_overwrite_guard(tmp_path, rng):
    vol, sino = rng.random((4, 4, 4)), rng.random((3, 4, 4))
    cache = SimCache(tmp_path)
    cache.save_run_volume("t", "xray", vol)
    cache.save_run_sinogram("t", "neutron", sino)
    np.testing.assert_array_equal(cache.load_run_volume("t", "xray"), _f32(vol))
    np.testing.assert_array_equal(cache.load_run_sinogram("t", "neutron"), _f32(sino))

    with pytest.raises(FileExistsError, match="overwrite=True"):
        cache.save_run_volume("t", "xray", vol + 1)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        cache.save_run_sinogram("t", "neutron", sino + 1)
    np.testing.assert_array_equal(cache.load_run_volume("t", "xray"), _f32(vol))

    writer = SimCache(tmp_path, overwrite=True)
    writer.save_run_volume("t", "xray", vol + 1)
    writer.save_run_sinogram("t", "neutron", sino + 1)
    np.testing.assert_array_equal(cache.load_run_volume("t", "xray"), _f32(vol + 1))
    np.testing.assert_array_equal(cache.load_run_sinogram("t", "neutron"), _f32(sino + 1))


def test_missing_run_data(tmp_path):
    cache = SimCache(tmp_path)
    with pytest.raises(FileNotFoundError, match="No saved volume for tag='x'"):
        cache.load_run_volume("x", "xray")
    with pytest.raises(FileNotFoundError, match="No sinogram for tag='x'"):
        cache.load_run_sinogram("x", "neutron")
    assert not (tmp_path / "runs").exists()  # failed reads leave no trace


# ── Inventory ────────────────────────────────────────────────────────────────

def test_list_run_tags_follows_save_order(tmp_path, tiny_phantom, rng):
    cache = SimCache(tmp_path)
    assert cache.list_run_tags() == []
    tags = ["Zeta run", "alpha (ref)", "Noise moderate\n(I₀=5×10⁴)"]
    for i, tag in enumerate(tags):
        cache.save_run(_result(tag, tiny_phantom, rng))
        # Ordering is by directory mtime; stamp distinct times explicitly so the
        # test does not depend on the file system's timestamp resolution.
        os.utime(cache.run_dir(tag), (1_000_000 + i, 1_000_000 + i))
    # Original tags (not slugs) come back, in save order rather than by name.
    assert cache.list_run_tags() == tags

    # A run directory without run_meta.json (e.g. a lone volume) is not a run.
    cache.save_run_volume("partial", "xray", np.zeros((2, 2, 2)))
    assert cache.list_run_tags() == tags
    assert not cache.has_run("partial")
    assert repr(cache) == f"SimCache(root='{tmp_path}', runs=3, overwrite=False)"


def test_has_run_and_clear_run(tmp_path, tiny_phantom, rng):
    cache = SimCache(tmp_path)
    for tag in ("keep", "drop"):
        cache.save_run(_result(tag, tiny_phantom, rng))
    assert cache.has_run("keep") and cache.has_run("drop")
    assert cache.has_run("DROP")  # lookup goes through the slug

    cache.clear_run("drop")
    assert not cache.has_run("drop") and cache.has_run("keep")
    assert cache.list_run_tags() == ["keep"]
    with pytest.raises(FileNotFoundError):
        cache.load_run_volume("drop", "xray")
    cache.clear_run("never saved")  # no-op
    assert cache.list_run_tags() == ["keep"]


def test_clear_all_removes_the_cache(tmp_path, tiny_phantom, rng):
    cache = SimCache(tmp_path / "cache")
    cache.save_phantom(tiny_phantom)
    cache.save_run(_result("r", tiny_phantom, rng))
    cache.clear_all()
    assert not cache.root.exists()
    assert cache.list_run_tags() == [] and not cache.has_run("r")
    # A new SimCache on the same path starts empty.
    assert not SimCache(tmp_path / "cache").phantom_exists()


# ── Phantom metadata and mismatch detection ──────────────────────────────────

def test_phantom_meta_and_guard(tmp_path, asymmetric_phantom):
    cache = SimCache(tmp_path)
    assert not cache.phantom_exists()
    cache.save_phantom(asymmetric_phantom)
    assert cache.phantom_exists()
    meta = cache.load_phantom_meta()
    assert meta["shape"] == [40, 40, 40] and meta["voxel_cm"] == 0.02
    assert meta["materials"] == ["Air", "Iron", "Water"]
    assert meta["material_keys"] == ["air", "iron", "water"]
    assert sorted(p.name for p in cache.phantom_dir.glob("mu_x_vol_*keV.npy"))[:2] == [
        "mu_x_vol_020keV.npy", "mu_x_vol_030keV.npy"]
    assert len(list(cache.phantom_dir.glob("mu_x_vol_*keV.npy"))) == len(nxs.XRAY_E_KEV)

    with pytest.raises(FileExistsError):
        cache.save_phantom(asymmetric_phantom)
    SimCache(tmp_path, overwrite=True).save_phantom(asymmetric_phantom)


def test_check_phantom_detects_a_different_simulation(tmp_path, asymmetric_phantom):
    cache = SimCache(tmp_path)
    cache.save_phantom(asymmetric_phantom)
    cache.check_phantom(asymmetric_phantom)  # same phantom: silent

    def variant(name="asymmetric", N=40, voxel_cm=0.02):
        b = nxs.PhantomBuilder(N=N, voxel_cm=voxel_cm)
        b.add_sphere("iron", radius_cm=0.1)
        return b.build(name)

    for other in (variant(name="other"), variant(N=36), variant(voxel_cm=0.03)):
        with pytest.raises(ValueError, match="holds phantom 'asymmetric'"):
            cache.check_phantom(other)


def test_modified_attenuation_volumes_round_trip(tmp_path, tiny_phantom):
    # e.g. the Bragg-edge scan rewrites mu_n_* in place; the cache must keep it.
    tiny_phantom.mu_n_abs_vol = tiny_phantom.mu_n_abs_vol * 3 + 0.5
    tiny_phantom.mu_n_vol = (tiny_phantom.mu_n_abs_vol + tiny_phantom.mu_n_coh_vol
                             + tiny_phantom.mu_n_inc_vol)
    cache = SimCache(tmp_path)
    cache.save_phantom(tiny_phantom)
    back = cache.load_phantom()
    np.testing.assert_array_equal(back.mu_n_abs_vol, tiny_phantom.mu_n_abs_vol)
    np.testing.assert_allclose(back.mu_n_vol, tiny_phantom.mu_n_vol, rtol=1e-6)
    np.testing.assert_array_equal(back.mu_x_vols, tiny_phantom.mu_x_vols)


def test_custom_material_must_be_registered_to_load(tmp_path, monkeypatch):
    pmma = nxs.material_from_formula("PMMA", "PMMA", "C5H8O2", density_gcc=1.18)
    b = nxs.PhantomBuilder(N=6, voxel_cm=0.1)
    b.add_sphere(pmma, radius_cm=0.2)
    cache = SimCache(tmp_path)
    cache.save_phantom(b.build("pmma"))
    with pytest.raises(KeyError, match="register_material"):
        cache.load_phantom()
    monkeypatch.setitem(nxs.MATERIALS, "test_pmma", pmma)  # restored after the test
    assert cache.load_phantom().materials[1] is pmma


def test_legacy_phantom_meta_without_keys_or_shape(tmp_path, tiny_phantom):
    # Caches written before material keys / shape were recorded still load.
    cache = SimCache(tmp_path)
    cache.save_phantom(tiny_phantom)
    path = cache.phantom_dir / "meta.json"
    meta = json.loads(path.read_text())
    del meta["material_keys"], meta["shape"]
    path.write_text(json.dumps(meta))
    back = cache.load_phantom()  # materials found by name instead
    assert back.materials == tiny_phantom.materials
    cache.check_phantom(tiny_phantom)  # shape falls back to N³


# ── Raw sinograms ────────────────────────────────────────────────────────────

def test_raw_sinograms_round_trip(tmp_path, tiny_phantom):
    xs, ns = nxs.make_sinogram_pair(tiny_phantom, n_angles=6, use_astra=False,
                                    verbose=False)
    cache = SimCache(tmp_path)
    assert not cache.raw_sinograms_exist()
    cache.save_raw_sinograms(xs, ns, params={"n_angles": 6})
    assert cache.raw_sinograms_exist()

    x = cache.load_raw_xray_sino()
    for key in ("sino_lam", "sino_trans"):
        np.testing.assert_array_equal(x[key], _f32(xs[key]))
    np.testing.assert_allclose(x["angles_deg"], xs["angles_deg"], rtol=1e-6)
    assert (x["voxel_cm"], x["I0"]) == (xs["voxel_cm"], xs["I0"])

    n = cache.load_raw_neutron_sino()
    for key in ("sino_lam", "sino_trans", "sino_abs_lam", "sino_scatter_lam"):
        np.testing.assert_array_equal(n[key], _f32(ns[key]))
    assert n["scatter_D_over_L"] == ns["scatter_D_over_L"]

    meta = json.loads((cache.sino_dir / "sino_meta.json").read_text())
    assert meta["kVp"] == xs["spectrum"]["kVp"]
    assert meta["sino_shape"] == list(xs["sino_lam"].shape) and meta["n_angles"] == 6

    with pytest.raises(FileExistsError):
        cache.save_raw_sinograms(xs, ns)


def test_raw_sinogram_optional_neutron_components_default(tmp_path, rng):
    lam = rng.random((3, 4, 4))
    xs = {"sino_lam": lam, "sino_trans": np.exp(-lam), "angles_deg": np.arange(3.0)}
    ns = {"sino_lam": 2 * lam, "sino_trans": np.exp(-2 * lam), "angles_deg": np.arange(3.0)}
    cache = SimCache(tmp_path)
    cache.save_raw_sinograms(xs, ns)
    n = cache.load_raw_neutron_sino()
    # No absorption/scatter split given: all attenuation counts as absorption.
    np.testing.assert_array_equal(n["sino_abs_lam"], _f32(2 * lam))
    assert not n["sino_scatter_lam"].any()
    assert (n["I0"], n["voxel_cm"], n["scatter_D_over_L"]) == (1e5, 0.0, 100.0)


def test_raw_sinograms_match_detects_stale_parameters(tmp_path, rng):
    lam = rng.random((2, 3, 3))
    sino = {"sino_lam": lam, "sino_trans": np.exp(-lam), "angles_deg": np.arange(2.0)}
    params = {"n_angles": 2, "kVp": 120.0, "phantom_shape": [3, 3, 3]}
    cache = SimCache(tmp_path)
    cache.save_raw_sinograms(sino, sino, params=params)

    assert cache.raw_sinograms_match(params)
    assert cache.raw_sinograms_match({**params, "new_param": 1})  # unknown keys ignored
    with pytest.warns(UserWarning, match=r"different parameters.*'kVp': \(120\.0, 90\.0\)"):
        assert not cache.raw_sinograms_match({**params, "kVp": 90.0})

    legacy = SimCache(tmp_path / "legacy")  # written before params were recorded
    legacy.save_raw_sinograms(sino, sino)
    assert legacy.raw_sinograms_match({"n_angles": 999})


def test_overwrite_cache_replaces_stale_sinograms(tmp_path):
    kw = dict(preset="composite", N=12, use_astra=False, verbose=False, cache_dir=tmp_path)
    nxs.DualModalitySimulation(n_angles=6, **kw).run(tag="first")
    with pytest.warns(UserWarning, match="different parameters"):
        sim = nxs.DualModalitySimulation(n_angles=10, overwrite_cache=True, **kw)
    sim.run(tag="second")
    cache = SimCache(tmp_path)
    assert cache.load_raw_xray_sino()["sino_lam"].shape[0] == 10
    assert cache.raw_sinograms_match(sim._projection_params())
    assert sorted(cache.list_run_tags()) == ["first", "second"]  # mtime order may tie


# ── Survey metrics ───────────────────────────────────────────────────────────

def _quality(ce, db, n_matched=2):
    return nxs.ClusterQualityMetrics(
        centroid_errors=ce, sigma_x={k: v / 2 for k, v in ce.items()},
        sigma_n={k: v / 3 for k, v in ce.items()},
        mean_centroid_error=float(np.mean(list(ce.values()))), davies_bouldin=db,
        overlap_fractions={}, n_matched=n_matched, gmm=None)


def test_survey_metrics_csv_round_trip(tmp_path):
    # Real survey tags contain newlines and non-ASCII characters.
    metrics = {
        "Clean (reference)": _quality({"Aluminum": 0.0123456789, "Iron": 0.2}, 0.31),
        "Noise moderate\n(I₀=5×10⁴)": _quality({"Aluminum": 0.05, "Iron": 1 / 3}, 0.62),
    }
    cache = SimCache(tmp_path)
    cache.save_survey_metrics("FBP_N32", metrics)
    rows = cache.load_survey_metrics_csv("FBP_N32")
    assert [r["tag"] for r in rows] == list(metrics)
    assert [r["slug"] for r in rows] == ["clean_reference", "noise_moderate_i_5_10"]
    for row, m in zip(rows, metrics.values()):
        assert float(row["mean_centroid_error"]) == m.mean_centroid_error
        assert float(row["davies_bouldin"]) == m.davies_bouldin
        assert int(row["n_matched"]) == m.n_matched
        for mat in ("Aluminum", "Iron"):  # str(float) round-trips exactly
            assert float(row[f"ce_{mat}"]) == m.centroid_errors[mat]
            assert float(row[f"sx_{mat}"]) == m.sigma_x[mat]
            assert float(row[f"sn_{mat}"]) == m.sigma_n[mat]

    with pytest.raises(FileExistsError):
        cache.save_survey_metrics("FBP_N32", metrics)


def test_survey_metrics_edge_cases(tmp_path):
    cache = SimCache(tmp_path)
    with pytest.raises(FileNotFoundError, match="No metrics CSV"):
        cache.load_survey_metrics_csv("missing")
    cache.save_survey_metrics("empty", {})  # nothing to write
    with pytest.raises(FileNotFoundError):
        cache.load_survey_metrics_csv("empty")
    # davies_bouldin is NaN with fewer than two matched clusters.
    cache.save_survey_metrics("one", {"r": _quality({"Iron": 0.1}, float("nan"), 1)})
    assert np.isnan(float(cache.load_survey_metrics_csv("one")[0]["davies_bouldin"]))


# ── tag_to_slug ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tag, slug", [
    ("Clean (reference)", "clean_reference"),                 # docstring examples
    ("Noise moderate\n(I₀=5×10⁴)", "noise_moderate_i_5_10"),
    ("  --Ring artifacts!!  ", "ring_artifacts"),             # trimmed
    ("a  +  b", "a_b"),                                       # runs collapse to one _
    ("FBP_N0512", "fbp_n0512"),
    ("Größe", "gr_e"),                                        # non-ASCII is dropped
    ("!!!", "run"),                                           # nothing left → fallback
    ("", "run"),
])
def test_tag_to_slug(tag, slug):
    assert tag_to_slug(tag) == slug


def test_tag_to_slug_truncation():
    assert tag_to_slug("x" * 100) == "x" * 64
    assert tag_to_slug("abc def", max_len=5) == "abc_d"
    for tag in ("Clean (reference)", "Noise + BH + rings", "a" * 80):
        slug = tag_to_slug(tag)
        assert tag_to_slug(slug) == slug  # idempotent
        assert len(slug) <= 64 and slug == slug.lower()


# ── Suspected bugs (strict xfail: they start passing once fixed) ──────────────

@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: save_phantom casts label_vol to uint8 (io.py:235), but imported "
    "phantoms may use uint16 labels (> 256 materials); labels >= 256 wrap"))
def test_phantom_with_more_than_256_labels_round_trips(tmp_path):
    seg = np.arange(300).reshape(3, 10, 10)
    class_map = {i: "copper" if i == 299 else "aluminum" for i in range(300)}
    ph = nxs.phantom_from_array(seg, {"voxel_cm": 0.1, "class_map": class_map})
    assert ph.label_vol.dtype == np.uint16
    cache = SimCache(tmp_path)
    cache.save_phantom(ph)
    np.testing.assert_array_equal(cache.load_phantom().label_vol, ph.label_vol)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: two tags with the same slug share runs/<slug>/ and save_run never "
    "calls _guard (io.py:466-512), so the second run silently replaces the "
    "first, even with overwrite=False; loading the first tag returns the "
    "second run's data"))
def test_runs_whose_tags_share_a_slug_are_not_mixed_up(tmp_path, tiny_phantom, rng):
    a = _result("Clean (reference)", tiny_phantom, rng)
    b = _result("clean reference", tiny_phantom, rng)
    assert tag_to_slug(a.tag) == tag_to_slug(b.tag)
    cache = SimCache(tmp_path)  # overwrite=False
    cache.save_run(a)
    try:
        cache.save_run(b)
    except FileExistsError:
        return  # refusing the second save would be a valid fix
    assert sorted(cache.list_run_tags()) == sorted([a.tag, b.tag])
    np.testing.assert_array_equal(cache.load_run_volume(a.tag, "xray"), _f32(a.vol_xray))


@pytest.mark.xfail(strict=True, raises=FileNotFoundError, reason=(
    "BUG: phantom_dir / sino_dir mkdir() on every access without parents=True "
    "(io.py:174-184): existence checks create directories, and on a cache "
    "whose root was removed by clear_all() they raise instead of returning False"))
def test_existence_checks_are_side_effect_free(tmp_path):
    cache = SimCache(tmp_path / "cache")
    assert not cache.phantom_exists() and not cache.raw_sinograms_exist()
    cache.clear_all()
    assert not cache.phantom_exists() and not cache.raw_sinograms_exist()
    assert not cache.root.exists()


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "BUG: save_survey_metrics takes the CSV columns from the first row only and "
    "passes extrasaction='ignore' (io.py:673-676), so per-material columns "
    "missing from the first run are silently dropped for every other run"))
def test_survey_metrics_keep_materials_missing_from_first_row(tmp_path):
    cache = SimCache(tmp_path)
    cache.save_survey_metrics("s", {
        "a": _quality({"Aluminum": 0.1}, 0.2, 1),
        "b": _quality({"Aluminum": 0.2, "Iron": 0.3}, 0.4, 2),
    })
    rows = cache.load_survey_metrics_csv("s")
    assert float(rows[1]["ce_Iron"]) == 0.3
