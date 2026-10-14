"""compute_histogram_metrics / HistogramMetricsTable (``analysis.metrics_table``)."""

import csv
import functools
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

import neutron_xray_sim as nxs
import neutron_xray_sim.analysis.metrics_table as mt
from neutron_xray_sim.acquisition.artifacts import inject_sinogram_artifacts
from neutron_xray_sim.analysis.gmm import predict_gmm
from neutron_xray_sim.analysis.quality import davies_bouldin, ground_truth_positions

E_IDX = nxs.DEFAULT_GT_ENERGY_IDX


@pytest.fixture(scope="module")
def three_phase():
    """Al cylinder with an HDPE and an Fe rod: three well-separated clusters.

    Ground truth (μ_x @ 80 keV, μ_n): Al (0.278, 0.098), HDPE (0.167, 2.18),
    Fe (4.12, 1.16).  The closest pair, Al–HDPE, is 2.085 cm⁻¹ apart.
    """
    b = nxs.PhantomBuilder(N=24, voxel_cm=1.6 / 24)
    b.add_cylinder("aluminum", radius_cm=0.65)
    b.add_rod("hdpe", center_cm=(-0.25, 0.0), radius_cm=0.2)
    b.add_rod("iron", center_cm=(0.25, 0.0), radius_cm=0.2)
    return b.build("three_phase")


def _gt_hist(phantom, sigma, seed=0, bins=64, n_shift=0.0, x_scale=1.0, **kw):
    """Histogram of the exact phantom values plus isotropic Gaussian noise σ."""
    r = np.random.default_rng(seed)
    vx = phantom.mu_x_vols[E_IDX] + sigma * r.standard_normal(phantom.shape)
    vn = phantom.mu_n_vol + sigma * r.standard_normal(phantom.shape) + n_shift
    return nxs.compute_bimodal_histogram(x_scale * vx, vn, bins=bins,
                                         mask=phantom.label_vol > 0, **kw)


def _gmm(hist, means, covs, weights):
    """Hand-built GMM, with voxel labels predicted like fit_gmm does."""
    g = nxs.GMMFitResult(n_components=len(means), means=np.asarray(means, float),
                         covariances=np.asarray(covs, float),
                         weights=np.asarray(weights, float),
                         labels_flat=np.empty(0, np.int32), bic=0.0, aic=0.0)
    g.labels_flat = predict_gmm(np.column_stack([hist.vol_x_flat, hist.vol_n_flat]), g)
    return g


# ── Metric values and degradation ────────────────────────────────────────────

def test_cluster_metrics_track_injected_noise(three_phase):
    # Same standard-normal draws scaled by σ: only the noise level changes.
    _, _, gt = ground_truth_positions(three_phase, E_IDX)
    tables = {}
    for sigma in (0.05, 0.1, 0.2):
        t = nxs.compute_histogram_metrics(three_phase, _gt_hist(three_phase, sigma),
                                          gmm_kwargs={"n_init": 1})
        tables[sigma] = t
        # The fitted spreads are the sample std of ≥ 672 voxels per material:
        # relative standard error 1/sqrt(2n) ≤ 2.7 %, so 10 % is > 3 SE.
        for key in ("sigma_x_k", "sigma_n_k"):
            assert set(t.per_cluster[key]) == {"Al", "HDPE", "Fe"}
            for v in t.per_cluster[key].values():
                assert v == pytest.approx(sigma, rel=0.1)
        # Isotropic clusters → DB = mean_k max_j 2σ / d_kj (≈ 0.807 σ here).
        assert t.scalars["DB"] == pytest.approx(davies_bouldin(gt, [sigma] * 3), rel=0.1)
        # Only sampling error of the means remains (Rayleigh mean ≈ 0.04 σ).
        assert t.scalars["CE"] < 0.15 * sigma
        assert not t.pathologies
    ce = [tables[s].scalars["CE"] for s in (0.05, 0.1, 0.2)]
    db = [tables[s].scalars["DB"] for s in (0.05, 0.1, 0.2)]
    assert ce == sorted(ce) and db == sorted(db) and len(set(ce)) == 3


def test_overlap_fraction_matches_bayes_error(three_phase):
    sigma = 0.5
    hist = _gt_hist(three_phase, sigma)
    n_al, n_hdpe = (int((three_phase.label_vol == i).sum()) for i in (1, 2))
    # Al–HDPE are 2.085 cm⁻¹ apart: excluded by the default 2.0 neighbour cut-off.
    default = nxs.compute_histogram_metrics(three_phase, hist, gmm_kwargs={"n_init": 1})
    assert default.pairwise == {}
    t = nxs.compute_histogram_metrics(three_phase, hist, overlap_threshold=2.5,
                                      gmm_kwargs={"n_init": 1})
    assert list(t.pairwise) == [("Al", "HDPE")]

    # Two isotropic Gaussians with priors ∝ voxel counts: the optimal boundary is
    # shifted by σ² ln(n_Al / n_HDPE) / d toward the rare class.
    _, _, gt = ground_truth_positions(three_phase, E_IDX)
    d = float(np.linalg.norm(gt[0] - gt[1]))
    shift = sigma ** 2 * math.log(n_al / n_hdpe) / d
    phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))  # noqa: E731
    bayes = (n_al * phi(-(d / 2 + shift) / sigma)
             + n_hdpe * phi(-(d / 2 - shift) / sigma)) / (n_al + n_hdpe)
    # ≈ 66 misclassified of 6528 voxels: Poisson SE ≈ 0.0012, tolerance ≈ 3 SE.
    assert t.pairwise[("Al", "HDPE")] == pytest.approx(bayes, abs=0.004)


@pytest.fixture(scope="module")
def recon_tables(three_phase):
    """Metrics of FBP reconstructions at decreasing dose (N=24, 36 angles)."""
    xs, ns = nxs.make_sinogram_pair(three_phase, n_angles=36, use_astra=False,
                                    verbose=False)
    # Score only voxels at least one voxel away from an interface, so that
    # partial-volume mixing does not dominate the cluster shapes.
    interior = np.zeros(three_phase.shape, bool)
    for lab in range(1, len(three_phase.materials)):
        interior |= ndimage.binary_erosion(three_phase.label_vol == lab, border_value=1)
    tables = []
    for i0 in (None, 5e2, 1e2):
        x, n = xs, ns
        if i0 is not None:
            x, n = inject_sinogram_artifacts(xs, ns, nxs.ArtifactConfig.noise_only(i0),
                                             rng=np.random.default_rng(0))
        vx, vn = nxs.reconstruct_pair(x, n, use_astra=False, verbose=False)
        hist = nxs.compute_bimodal_histogram(vx, vn, bins=64, mask=interior)
        tables.append(nxs.compute_histogram_metrics(three_phase, hist,
                                                    gt_seeded_gmm=True))
    return tables


def test_photon_noise_degrades_reconstruction_metrics(recon_tables):
    clean, mid, low = recon_tables
    # DB and every neutron cluster width grow as the dose drops; over noise
    # seeds 0–9 the smallest step ratio was 1.05, so the order is not luck.
    # σ_x is not checked: the X-ray clusters are already broadened by beam
    # hardening (polychromatic projector), so the added noise is a small
    # relative change that a GMM re-assignment can mask (1 seed in 10).
    assert clean.scalars["DB"] < mid.scalars["DB"] < low.scalars["DB"]
    for mat in clean.per_cluster["sigma_n_k"]:
        s = [t.per_cluster["sigma_n_k"][mat] for t in recon_tables]
        assert s[0] < s[1] < s[2], (mat, s)
    for t in recon_tables:
        assert t.has_ground_truth and t.n_components == 3
        assert all(np.isfinite(v) for v in t.scalars.values() if v is not None)


# ── GMM seeding and pathology detection ──────────────────────────────────────

def test_gt_seeded_gmm_recovers_a_rare_phase():
    # 8 Al voxels among ~23 000: an unseeded 3-component GMM spends no component
    # on them; seeding at the ground-truth positions keeps the phase visible.
    b = nxs.PhantomBuilder(N=32, voxel_cm=0.05)
    b.add_cylinder("hdpe", radius_cm=0.75)
    b.add_box("water", center_cm=(0, 0, 0.3), half_extents_cm=(1, 0.5, 0.2))
    b.add_sphere("aluminum", center_cm=(0, -0.3, -0.3), radius_cm=0.08)
    ph = b.build("rare")
    assert (ph.label_vol == 3).sum() == 8
    hist = _gt_hist(ph, 0.05)

    plain = nxs.compute_histogram_metrics(ph, hist, gmm_kwargs={"n_init": 1})
    assert plain.per_cluster["eps_k"]["Al"] > 1.0
    assert [p["data"]["material"] for p in plain.pathologies
            if p["kind"] == "uncovered_ground_truth"] == ["Al"]
    with pytest.raises(RuntimeError, match="pathology"):
        nxs.compute_histogram_metrics(ph, hist, gmm=nxs.fit_gmm(hist, 3, n_init=1),
                                      raise_on_pathology=True)

    seeded = nxs.compute_histogram_metrics(ph, hist, gt_seeded_gmm=True,
                                           raise_on_pathology=True)
    assert max(seeded.per_cluster["eps_k"].values()) < 0.05
    # The rare component's weight (8 / 22 912) is reported, as info only.
    assert [p["kind"] for p in seeded.pathologies] == ["degenerate_component_weight"]
    assert seeded.pathologies[0]["severity"] == "info"


def test_pathologies_on_a_constructed_fit(three_phase):
    hist = _gt_hist(three_phase, 0.05)
    _, _, gt = ground_truth_positions(three_phase, E_IDX)
    al, fe = gt[0], gt[2]
    gmm = _gmm(hist, means=[al, al + [0.1, 0.0], fe + [0.0, 3.5]],
               covs=[np.eye(2) * 1e-2, np.eye(2) * 1e-2, np.diag([1.0, 1e-4])],
               weights=[0.996, 0.002, 0.002])
    t = nxs.compute_histogram_metrics(three_phase, hist, gmm=gmm)

    # Greedy matching: Al→0, then HDPE→1 (2.08 away), Fe→2 (3.5 away).
    eps = t.per_cluster["eps_k"]
    assert eps["Al"] == pytest.approx(0.0, abs=1e-12)
    assert eps["Fe"] == pytest.approx(3.5)
    assert eps["HDPE"] == pytest.approx(np.linalg.norm(gt[1] - gmm.means[1]))
    assert t.scalars["CE"] == pytest.approx(np.mean(list(eps.values())))
    assert t.per_cluster["sigma_x_k"]["Fe"] == pytest.approx(1.0)
    assert t.per_cluster["sigma_n_k"]["Fe"] == pytest.approx(1e-2)
    assert t.per_cluster["E_k"] == pytest.approx({"comp_0": 1.0, "comp_1": 1.0,
                                                  "comp_2": 100.0}, rel=1e-6)

    found = {}
    for p in t.pathologies:
        found.setdefault(p["kind"], []).append(p)
    assert set(found) == {"large_centroid_error", "degenerate_component_weight",
                          "elongated_components", "duplicate_components",
                          "uncovered_ground_truth"}
    assert found["large_centroid_error"][0]["data"]["material"] == "Fe"
    assert found["degenerate_component_weight"][0]["data"]["components"] == [1, 2]
    assert list(found["elongated_components"][0]["data"]) == [2]
    assert found["duplicate_components"][0]["data"]["pairs"] == [(0, 1)]
    assert {p["data"]["material"] for p in found["uncovered_ground_truth"]} == {"HDPE", "Fe"}
    text = str(t)
    assert "Pathology warnings" in text and "[WARNING]" in text and "[INFO   ]" in text

    with pytest.raises(RuntimeError, match="pathology warnings detected"):
        nxs.compute_histogram_metrics(three_phase, hist, gmm=gmm, raise_on_pathology=True)
    # Relaxed thresholds leave only the info-level findings, which never raise.
    relaxed = nxs.compute_histogram_metrics(
        three_phase, hist, gmm=gmm, raise_on_pathology=True,
        pathology_thresholds={"max_centroid_error_cm": 5, "max_elongation": 200,
                              "uncovered_gt_distance_cm": 5})
    assert {p["severity"] for p in relaxed.pathologies} == {"info"}
    off = nxs.compute_histogram_metrics(three_phase, hist, gmm=gmm,
                                        detect_pathologies=False)
    assert off.pathologies == [] and off.scalars == t.scalars


def test_ideal_fit_has_no_pathologies(three_phase):
    hist = _gt_hist(three_phase, 0.05)
    _, _, gt = ground_truth_positions(three_phase, E_IDX)
    gmm = _gmm(hist, means=gt, covs=[np.eye(2) * 0.05 ** 2] * 3,
               weights=[0.8, 0.1, 0.1])
    t = nxs.compute_histogram_metrics(three_phase, hist, gmm=gmm, overlap_threshold=2.5)
    assert t.scalars["CE"] == 0.0 and not t.pathologies
    assert t.scalars["DB"] == pytest.approx(davies_bouldin(gt, [0.05] * 3))
    assert t.pairwise == {("Al", "HDPE"): 0.0}  # 2.085 cm⁻¹ = 41 σ apart
    assert t.n_components == 3 and t.energy_idx == E_IDX


# ── Optional inputs and degenerate cases ─────────────────────────────────────

def test_reference_histogram_gives_neutron_shift(three_phase):
    ref = _gt_hist(three_phase, 0.05)
    shifted = _gt_hist(three_phase, 0.05, n_shift=0.25)  # same draws, μ_n + 0.25
    t = nxs.compute_histogram_metrics(three_phase, shifted, ref_hist=ref,
                                      gmm_kwargs={"n_init": 1})
    assert t.has_reference and t.scalars["Delta_n"] == pytest.approx(0.25, rel=1e-9)
    assert not any("Δ_n" in n for n in t.notes)
    no_ref = nxs.compute_histogram_metrics(three_phase, shifted, gmm_kwargs={"n_init": 1})
    assert no_ref.scalars["Delta_n"] is None and not no_ref.has_reference
    assert any("Δ_n" in n for n in no_ref.notes)


def test_normalised_streak_scores_ignore_axis_scale(three_phase):
    # With a fixed binning range, stretching μ_x changes the bin aspect ratio.
    kw = dict(x_range=(-0.5, 10.0), n_range=(-0.5, 3.0), bins=32)
    h1 = _gt_hist(three_phase, 0.05, **kw)
    h2 = _gt_hist(three_phase, 0.05, x_scale=2.0, **kw)
    run = functools.partial(nxs.compute_histogram_metrics, three_phase,
                            gmm_kwargs={"n_init": 1})
    raw1, raw2 = run(h1), run(h2)
    assert raw1.scalars["S_v"] != pytest.approx(raw2.scalars["S_v"], rel=0.01)
    n1, n2 = (run(h, normalize_shape_axes=True) for h in (h1, h2))
    for key in ("S_h", "S_v"):
        assert n1.scalars[key] == pytest.approx(n2.scalars[key], rel=1e-9)
    assert any("axis-normalised" in n for n in n1.notes)
    # Only S_h / S_v are affected by the option.
    assert n1.scalars["S_d"] == raw1.scalars["S_d"]


def test_without_label_volume_only_shape_metrics(three_phase):
    no_gt = SimpleNamespace(materials=three_phase.materials, label_vol=None)
    t = nxs.compute_histogram_metrics(no_gt, _gt_hist(three_phase, 0.05),
                                      gmm_kwargs={"n_init": 1})
    assert not t.has_ground_truth and "CE" not in t.scalars and t.pairwise == {}
    assert any("no label_vol" in n for n in t.notes)
    assert {r["metric"] for r in t.to_records()} == {"S_h", "S_v", "S_d", "A_x",
                                                     "Delta_n", "E_k"}
    assert "skipped — no ground truth" in str(t)


def test_histogram_off_the_phantom_grid_skips_overlap(three_phase):
    r = np.random.default_rng(0)
    vx = three_phase.mu_x_vols[E_IDX][:, :, :12] + 0.05 * r.standard_normal((24, 24, 12))
    vn = three_phase.mu_n_vol[:, :, :12] + 0.05 * r.standard_normal((24, 24, 12))
    hist = nxs.compute_bimodal_histogram(vx, vn, bins=32)
    t = nxs.compute_histogram_metrics(three_phase, hist, overlap_threshold=2.5,
                                      gmm_kwargs={"n_init": 1})
    assert t.pairwise == {} and any("not on the phantom grid" in n for n in t.notes)


def test_single_material_has_no_davies_bouldin():
    b = nxs.PhantomBuilder(N=12, voxel_cm=0.1)
    b.add_sphere("iron", radius_cm=0.4)
    ph = b.build("one")
    t = nxs.compute_histogram_metrics(ph, _gt_hist(ph, 0.05), gmm_kwargs={"n_init": 1})
    assert t.n_components == 1 and t.scalars["DB"] is None
    assert any("DB (2.5) needs" in n for n in t.notes)
    assert "N/A" in str(t)


def test_air_only_phantom_is_rejected():
    ph = nxs.PhantomBuilder(N=6, voxel_cm=0.1).build("empty")
    hist = nxs.compute_bimodal_histogram(ph.mu_x_vols[E_IDX], ph.mu_n_vol, bins=8)
    with pytest.raises(ValueError, match="No non-air materials"):
        nxs.compute_histogram_metrics(ph, hist)


def test_n_components_override(three_phase):
    t = nxs.compute_histogram_metrics(three_phase, _gt_hist(three_phase, 0.05),
                                      n_components=5, gmm_kwargs={"n_init": 1})
    assert t.n_components == 5 and len(t.per_cluster["E_k"]) == 5
    assert len(t.per_cluster["eps_k"]) == 3  # one match per ground-truth material


# ── Export ───────────────────────────────────────────────────────────────────

@pytest.fixture
def full_table(three_phase):
    """A table with every metric family: shape, Δ_n, per-cluster, pairwise."""
    return nxs.compute_histogram_metrics(
        three_phase, _gt_hist(three_phase, 0.5, n_shift=0.1),
        ref_hist=_gt_hist(three_phase, 0.5), overlap_threshold=2.5,
        gmm_kwargs={"n_init": 1})


def test_to_records_layout(full_table):
    t = full_table
    recs = t.to_records()
    cols = ["family", "name", "eq", "unit", "diagnoses", "metric", "cluster_or_pair",
            "value"]
    assert all(list(r) == cols for r in recs)
    metrics = [r["metric"] for r in recs]
    # 5 global shape metrics, K elongations, CE + DB, 3 per-material families, pairs.
    assert metrics == (["S_h", "S_v", "S_d", "A_x", "Delta_n"] + ["E_k"] * 3
                       + ["CE", "DB"] + ["eps_k"] * 3 + ["sigma_x_k"] * 3
                       + ["sigma_n_k"] * 3 + ["O_ab"])
    by_key = {(r["metric"], r["cluster_or_pair"]): r["value"] for r in recs}
    assert by_key[("Delta_n", "—")] == t.scalars["Delta_n"]
    assert by_key[("eps_k", "Fe")] == t.per_cluster["eps_k"]["Fe"]
    assert by_key[("O_ab", "Al ↔ HDPE")] == t.pairwise[("Al", "HDPE")]
    assert recs[0]["unit"] == "dimensionless" and recs[-1]["eq"] == "(2.7)"


def test_to_dataframe_matches_records(full_table):
    pd = pytest.importorskip("pandas")
    df = full_table.to_dataframe()
    assert isinstance(df, pd.DataFrame) and len(df) == len(full_table.to_records())
    assert df.to_dict("records") == full_table.to_records()


def test_to_dataframe_without_pandas(full_table, monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="pandas is required"):
        full_table.to_dataframe()


def test_to_csv_round_trip(full_table, tmp_path):
    path = tmp_path / "metrics.csv"
    full_table.to_csv(str(path))
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    recs = full_table.to_records()
    assert len(rows) == len(recs)
    for row, rec in zip(rows, recs):
        assert {k: row[k] for k in ("metric", "cluster_or_pair", "unit", "name")} == {
            k: rec[k] for k in ("metric", "cluster_or_pair", "unit", "name")}
        assert float(row["value"]) == rec["value"]  # repr(float) round-trips exactly


def test_to_csv_writes_missing_values_as_empty(three_phase, tmp_path):
    t = nxs.compute_histogram_metrics(three_phase, _gt_hist(three_phase, 0.05),
                                      gmm_kwargs={"n_init": 1})
    t.to_csv(tmp_path / "m.csv")
    with open(tmp_path / "m.csv", newline="", encoding="utf-8") as f:
        delta = [r for r in csv.DictReader(f) if r["metric"] == "Delta_n"]
    assert delta[0]["value"] == ""


def test_empty_table_export(tmp_path):
    t = nxs.HistogramMetricsTable()
    assert t.to_records() == []
    with pytest.raises(ValueError, match="No metrics"):
        t.to_csv(str(tmp_path / "m.csv"))
    assert not (tmp_path / "m.csv").exists()
    text = str(t)
    assert "ground truth: no" in text and "N/A" in text and repr(t) == text


# ── Suspected bugs (strict xfail: they start passing once fixed) ──────────────

@pytest.mark.xfail(strict=True, raises=IndexError, reason=(
    "BUG: ground-truth positions are keyed by material symbol (metrics_table.py:593, "
    "642), so two labels mapped to the same material leave gt_names shorter than "
    "gt_pts and gt_names[g] at line 644 raises IndexError; such phantoms are "
    "legal importer output and evaluate_histogram_quality accepts them"))
def test_two_labels_with_the_same_material():
    seg = np.zeros((16, 16, 16), int)
    seg[:, 2:8] = 1
    seg[:, 8:14] = 2
    seg[:, :, 10:14] = 3
    ph = nxs.phantom_from_array(seg, {"voxel_cm": 0.05, "class_map": {
        0: "air", 1: "aluminum", 2: "aluminum", 3: "iron"}})
    t = nxs.compute_histogram_metrics(ph, _gt_hist(ph, 0.02), gmm_kwargs={"n_init": 1})
    assert set(t.per_cluster["eps_k"]) == {"Al", "Fe"}


# Regression: to_csv used the locale encoding, so it raised on Windows (cp1252).
def test_to_csv_does_not_depend_on_the_locale(full_table, tmp_path, monkeypatch):
    # Emulate a cp1252 default; an explicit encoding= in to_csv still wins.
    monkeypatch.setattr(mt, "open", functools.partial(open, encoding="cp1252"),
                        raising=False)
    full_table.to_csv(tmp_path / "m.csv")
    assert "cm⁻¹" in (tmp_path / "m.csv").read_text(encoding="utf-8")
