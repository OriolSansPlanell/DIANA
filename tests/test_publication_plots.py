"""Figure functions: the top-level histogram figures and the publication set
(``plotting.publication``, formerly ``diana_plots``).

The publication functions return None: each saves one PDF into ``output_dir``
and closes its figure.  ``saved_figures`` captures the figure at save time so
its content can be checked.
"""

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure

import neutron_xray_sim as nxs
from neutron_xray_sim.plotting import publication as pub

E_IDX = nxs.DEFAULT_GT_ENERGY_IDX
N_PROJ = [16, 32, 64]


@pytest.fixture(autouse=True)
def _isolated_figures(monkeypatch, tmp_path):
    """Close figures, check rcParams are untouched, and work in tmp_path so a
    stray save to the default OUTPUT_DIR (a relative path) cannot reach the repo."""
    monkeypatch.chdir(tmp_path)
    before = mpl.rcParams.copy()
    yield
    plt.close("all")
    assert mpl.rcParams == before, "a plotting function changed global rcParams"


@pytest.fixture
def out(tmp_path):
    return tmp_path / "figures"


@pytest.fixture
def saved_figures(monkeypatch):
    figs = []
    original = Figure.savefig

    def spy(self, fname, *args, **kwargs):
        figs.append(self)
        return original(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    return figs


def _pdfs(out):
    """Names of the files written, after checking each is a non-empty PDF."""
    files = sorted(pathlib.Path(out).iterdir())
    for f in files:
        assert f.read_bytes()[:5] == b"%PDF-", f
    return [f.name for f in files]


@pytest.fixture
def hist():
    """Two-cluster bimodal histogram with empty bins."""
    r = np.random.default_rng(0)
    vx = np.concatenate([r.normal(0.3, 0.03, 600), r.normal(4.1, 0.1, 300)])
    vn = np.concatenate([r.normal(0.1, 0.02, 600), r.normal(1.2, 0.05, 300)])
    return nxs.compute_bimodal_histogram(vx, vn, bins=16)


@pytest.fixture
def phantom_hist(asymmetric_phantom):
    r = np.random.default_rng(0)
    ph = asymmetric_phantom
    vx = ph.mu_x_vols[E_IDX] + 0.02 * r.standard_normal(ph.shape)
    vn = ph.mu_n_vol + 0.02 * r.standard_normal(ph.shape)
    return nxs.compute_bimodal_histogram(vx, vn, bins=32, mask=ph.label_vol > 0)


# ── Top-level figure functions (return a Figure) ─────────────────────────────

def test_plot_bimodal_histogram(phantom_hist):
    gmm = nxs.fit_gmm(phantom_hist, n_components=2, n_init=1)
    fig = nxs.plot_bimodal_histogram(phantom_hist, gmm=gmm, title="T",
                                     material_labels={0: "first"})
    assert isinstance(fig, plt.Figure)
    ax = next(a for a in fig.axes if a.get_title() == "T")
    assert len([p for p in ax.patches if type(p).__name__ == "Ellipse"]) == 2
    labels = {t.get_text() for t in ax.texts}
    assert labels == {"first", "#1"}

    # Into an existing axes: no marginals, returns that axes' figure.
    host, host_ax = plt.subplots()
    assert nxs.plot_bimodal_histogram(phantom_hist, ax=host_ax, log_scale=False) is host
    assert host_ax.get_images()[0].get_extent() == pytest.approx(phantom_hist.extent)


def test_plot_comparison_grid_hides_unused_panels(phantom_hist):
    fig = nxs.plot_comparison_grid([(f"run {i}", phantom_hist) for i in range(4)],
                                   ncols=3, suptitle="Survey")
    assert isinstance(fig, plt.Figure) and fig._suptitle.get_text() == "Survey"
    panels = [a for a in fig.axes if a.get_title().startswith("run")]
    assert [a.get_title() for a in panels] == ["run 0", "run 1", "run 2", "run 3"]
    assert sum(not a.get_visible() for a in fig.axes) == 2  # 2×3 grid, 4 used


def test_plot_ground_truth_comparison(asymmetric_phantom, phantom_hist):
    fig = nxs.plot_ground_truth_comparison(asymmetric_phantom, phantom_hist)
    assert isinstance(fig, plt.Figure)
    gt_ax = next(a for a in fig.axes if a.get_title() == "Ground Truth")
    # One bubble per material at its exact (μ_x, μ_n).
    bubbles = [c.get_offsets()[0] for c in gt_ax.collections if len(c.get_offsets())]
    expected = [(m._mu_x_table[E_IDX], m.mu_n) for m in asymmetric_phantom.materials]
    np.testing.assert_allclose(bubbles, expected)
    recon = next(a for a in fig.axes if a.get_title() == "Reconstructed")
    # Both panels share the axis limits.
    assert recon.get_xlim() == gt_ax.get_xlim() and recon.get_ylim() == gt_ax.get_ylim()


def test_plot_artifact_survey(asymmetric_phantom, phantom_hist):
    run = nxs.SimulationResult(
        tag="noise", cfg=nxs.ArtifactConfig.clean(), phantom=asymmetric_phantom,
        vol_xray=None, vol_neutron=None, xray_sino={}, neutron_sino={},
        histogram=phantom_hist)
    metrics = {"noise": nxs.ClusterQualityMetrics(
        centroid_errors={}, sigma_x={}, sigma_n={}, mean_centroid_error=0.0123,
        davies_bouldin=float("nan"), overlap_fractions={}, n_matched=0, gmm=None)}
    fig = nxs.plot_artifact_survey({"noise": run}, asymmetric_phantom, metrics=metrics)
    titles = [a.get_title() for a in fig.axes]
    assert "Ground Truth\n(exact positions)" in titles
    assert "noise\nDB=n/a  CE=0.0123 cm⁻¹" in titles


def test_plot_cross_algorithm_grid():
    ph = nxs.make_phantom("composite", N=12)
    xs, ns = nxs.make_cross_algorithm_sinos(ph, ["FBP"], n_angles=8, use_astra=False)
    fig, hists = nxs.plot_cross_algorithm_grid(ph, xs, ns, [("FBP", "FBP")], bins=16,
                                               show_metrics=False, use_astra=False)
    assert isinstance(fig, plt.Figure) and list(hists) == [("FBP", "FBP")]
    assert hists[("FBP", "FBP")].H.shape == (16, 16)
    assert "X-ray: FBP  |  Neutron: FBP" in [a.get_title() for a in fig.axes]


def test_plot_spectra():
    beams = {"thermal": nxs.thermal_beam(), "ILL": nxs.ill_next_beam()}
    fig = nxs.plot_spectra(beams)
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 2
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "thermal (25.3 meV)", "ILL"]


# ── Publication figure set (save one PDF each, return None) ──────────────────

def test_plot_phantom_label_map(out, saved_figures):
    label = np.array([[0, 1, 1], [2, 1, 0]], np.uint8)
    mats = [nxs.AIR, nxs.ALUMINUM, nxs.IRON]
    assert pub.plot_phantom_label_map(label, mats, voxel_size_mm=0.5, geo_label="G7",
                                      output_dir=out) is None
    assert _pdfs(out) == ["G7_phantom_label_map.pdf"]
    fig = saved_figures[-1]
    im = fig.axes[0].get_images()[0]
    assert im.get_extent() == [0, 1.5, 0, 1.0]  # (nx, ny) × voxel size in mm
    # MAT_COLORS wins by symbol ('Air'); other materials keep their own colour.
    assert list(im.cmap.colors) == [pub.MAT_COLORS["Air"], nxs.ALUMINUM.color,
                                    nxs.IRON.color]
    assert [t.get_text() for t in fig.axes[1].get_yticklabels()] == [
        "[0] Air", "[1] Al", "[2] Fe"]
    assert plt.get_fignums() == []  # closed after saving


def test_publication_bimodal_histogram(hist, out, saved_figures):
    table = nxs.HistogramMetricsTable(scalars={"CE": 0.01234, "DB": None})
    pub.plot_bimodal_histogram(hist, table=table, n_proj=64, condition="dirty",
                               output_dir=out)
    pub.plot_bimodal_histogram(hist, output_dir=out)
    assert _pdfs(out) == ["hist_N0064_dirty.pdf", "hist_Nall_clean.pdf"]
    main, top, right, _ = saved_figures[0].axes
    assert "CE=0.0123 cm⁻¹" in top.get_title() and "DB=" not in top.get_title()
    # Empty bins are masked; the marginals show only non-empty bins.
    assert np.ma.count_masked(main.collections[0].get_array()) == (hist.H == 0).sum()
    x_marg = hist.H.sum(axis=1)
    assert [p.get_height() for p in top.patches] == list(x_marg[x_marg > 0])
    assert main.get_xlim() == pytest.approx(hist.x_edges[[0, -1]])


def test_marginal_wrappers_are_no_ops(hist, out):
    assert pub.plot_xray_marginal(hist, output_dir=out) is None
    assert pub.plot_neutron_marginal(hist, output_dir=out) is None
    assert not out.exists()


@pytest.mark.parametrize("modality, cmap", [("xray", "bone"), ("neutron", "viridis")])
def test_plot_reconstruction_slice(modality, cmap, out, saved_figures):
    vol = np.linspace(0, 2, 15).reshape(3, 5)
    pub.plot_reconstruction_slice(vol, modality=modality, n_proj=8, voxel_size_mm=0.5,
                                  output_dir=out)
    assert _pdfs(out) == [f"G1_recon_{modality}_N0008_clean.pdf"]
    im = saved_figures[-1].axes[0].get_images()[0]
    assert im.get_cmap().name == cmap
    assert im.get_clim() == (0, 2.0)  # vmax defaults to the data maximum
    assert im.get_extent() == [0, 2.5, 0, 1.5]


@pytest.mark.parametrize("func, stem", [
    (pub.plot_CE_vs_nprojections, "CE"),
    (pub.plot_DB_vs_nprojections, "DB"),
])
def test_scalar_vs_nprojections(func, stem, out, saved_figures):
    func(N_PROJ, [0.3, 0.2, 0.1], geo_label="G2", condition="dirty", output_dir=out)
    assert _pdfs(out) == [f"G2_{stem}_vs_Nproj_dirty.pdf"]
    ax = saved_figures[-1].axes[0]
    np.testing.assert_array_equal(ax.get_lines()[0].get_ydata(), [0.3, 0.2, 0.1])
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["16", "32", "64"]


@pytest.mark.parametrize("func, stem", [
    (pub.plot_sigma_x_vs_nprojections, "sigma_x"),
    (pub.plot_sigma_n_vs_nprojections, "sigma_n"),
    (pub.plot_epsilon_vs_nprojections, "epsilon"),
])
def test_per_material_vs_nprojections(func, stem, out, saved_figures):
    data = {"HAp": [0.3, 0.2, 0.1], "Fe": [0.5, 0.4, 0.3]}
    func(N_PROJ, data, geo_label="G3", output_dir=out)
    assert _pdfs(out) == [f"G3_{stem}_vs_Nproj_clean.pdf"]
    ax = saved_figures[-1].axes[0]
    for line, values in zip(ax.get_lines(), data.values()):
        np.testing.assert_array_equal(line.get_ydata(), values)
    # MAT_COLORS palette first, grey for symbols it does not know.
    assert [line.get_color() for line in ax.get_lines()] == ["#E8B87A", "#555555"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["HAp", "Fe"]


def test_plot_pairwise_overlap_vs_nprojections(out, saved_figures):
    data = {("HAp", "Hem"): [0.4, 0.2, 0.1], ("Org", "Qtz"): [0.3, 0.2, 0.0]}
    pub.plot_pairwise_overlap_vs_nprojections(N_PROJ, data, output_dir=out)
    assert _pdfs(out) == ["G1_pairwise_overlap_vs_Nproj_clean.pdf"]
    ax = saved_figures[-1].axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["HAp–Hem", "Org–Qtz"]
    assert ax.get_ylim()[0] == 0


@pytest.mark.parametrize("func, metric", [
    (pub.plot_CE_all_geometries, "CE"),
    (pub.plot_DB_all_geometries, "DB"),
])
def test_all_geometries_one_pdf_each(func, metric, out, saved_figures):
    table = lambda ce, db: nxs.HistogramMetricsTable(scalars={"CE": ce, "DB": db})  # noqa: E731
    sweep = {
        "G2_multi": {"clean": {n: {"table_la": table(0.1, 0.5)} for n in N_PROJ},
                     "dirty": {n: {"table_la": table(0.3, 0.9)} for n in N_PROJ}},
        # Missing tables and missing values become NaN; absent conditions are skipped.
        "G1_disc": {"clean": {16: {"table_la": None}, 32: {"table_la": table(None, None)},
                              64: {"table_la": table(0.2, 0.4)}}},
    }
    func(sweep, N_PROJ, output_dir=out)
    assert _pdfs(out) == [f"G1_disc_{metric}_vs_Nproj_clean_vs_dirty.pdf",
                          f"G2_multi_{metric}_vs_Nproj_clean_vs_dirty.pdf"]
    disc, multi = saved_figures  # sorted(geo_keys) order
    expected = 0.2 if metric == "CE" else 0.4
    np.testing.assert_array_equal(disc.axes[0].get_lines()[0].get_ydata(),
                                  [np.nan, np.nan, expected])
    assert [t.get_text() for t in disc.axes[0].get_legend().get_texts()] == ["Clean"]
    assert [t.get_text() for t in multi.axes[0].get_legend().get_texts()] == [
        "Clean", "Dirty"]


@pytest.mark.parametrize("func, kwargs, fname", [
    (pub.plot_cross_sweep_heatmap, {"projection_counts": [8, 16, 32]}, "cross_sweep_DB.pdf"),
    (pub.plot_algorithm_cross_heatmap, {"algorithms": ["FBP", "SIRT", "CGLS"]},
     "cross_algorithm_DB.pdf"),
])
def test_cross_heatmaps(func, kwargs, fname, out, saved_figures):
    m = np.array([[0.1, 0.2, np.nan], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
    func(m, metric="DB", good_is_low=False, output_dir=out, **kwargs)
    assert _pdfs(out) == [fname]
    ax = saved_figures[-1].axes[0]
    cells = {t.get_position(): t for t in ax.texts}  # (column, row)
    assert cells[(2, 0)].get_text() == "—"  # NaN cell
    assert cells[(1, 2)].get_text() == "0.700"
    # The matched diagonal is bold and outlined.
    assert {p for p, t in cells.items() if t.get_fontweight() == "bold"} == {
        (0, 0), (1, 1), (2, 2)}
    assert len(ax.patches) == 3
    assert ax.get_images()[0].get_clim() == (0.1, 0.8)  # NaN-aware colour range


def test_plot_xray_attenuation_spectra_skips_air(out, saved_figures):
    pub.plot_xray_attenuation_spectra([nxs.AIR, nxs.ALUMINUM, nxs.IRON], nxs.XRAY_E_KEV,
                                      output_dir=out)
    assert _pdfs(out) == ["xray_attenuation_spectra.pdf"]
    ax = saved_figures[-1].axes[0]
    lines = ax.get_lines()
    assert len(lines) == 2
    np.testing.assert_allclose(lines[1].get_ydata(), nxs.IRON._mu_x_table)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Al (ρ=2.70 g/cc)", "Fe (ρ=7.87 g/cc)"]


def test_plot_artifact_fingerprint(out, saved_figures):
    z = np.array([[0.5, -2.0], [np.nan, 1.0]])
    pub.plot_artifact_fingerprint(z, ["noise", "rings"], ["horizontal_streak_score", "S_d"],
                                  file_name="fp", output_dir=out)
    assert _pdfs(out) == ["fp.pdf"]
    ax = saved_figures[-1].axes[0]
    assert [t.get_text() for t in ax.texts] == ["0.50", "-2.00", "1.00"]  # NaN unlabelled
    assert ax.get_images()[0].get_clim() == (-2.0, 2.0)  # symmetric about z = 0
    assert [t.get_text() for t in ax.get_xticklabels()] == ["horizontal\nstreak", "S\nd"]


def test_plot_metric_by_artifact(out, saved_figures):
    pub.plot_metric_by_artifact([0.1, 0.3], ["noise", "rings"], metric="CE",
                                baseline=0.05, output_dir=out)
    pub.plot_metric_by_artifact([1.0], ["noise"], metric="DB", file_name="db",
                                output_dir=out)
    assert _pdfs(out) == ["db.pdf", "metric_CE_by_artifact.pdf"]
    ax = saved_figures[0].axes[0]
    assert [p.get_width() for p in ax.patches] == [0.1, 0.3]
    assert ax.get_lines()[0].get_xdata() == [0.05, 0.05]
    assert "cm" in ax.get_xlabel() and "cm" not in saved_figures[1].axes[0].get_xlabel()


def test_plot_rare_phase_failure_orders_by_fraction(out, saved_figures):
    pub.plot_rare_phase_failure([0.01, 0.1, 0.001], [1, 2, 3], [4, 5, 6],
                                ratio=[7, 8, 9], output_dir=out)
    assert _pdfs(out) == ["rare_phase_failure.pdf"]
    ax, ax2 = saved_figures[-1].axes
    v1, v3 = ax.get_lines()
    np.testing.assert_array_equal(v1.get_xdata(), [0.1, 0.01, 0.001])  # common → rare
    np.testing.assert_array_equal(v1.get_ydata(), [2, 1, 3])
    np.testing.assert_array_equal(v3.get_ydata(), [5, 4, 6])
    np.testing.assert_array_equal(ax2.get_lines()[0].get_ydata(), [8, 7, 9])
    assert ax.xaxis_inverted() and len(ax.get_legend().get_texts()) == 3


def test_plot_metric_vs_param_multi(out, saved_figures):
    pub.plot_metric_vs_param_multi([1, 2, 4], {"a": [1, 2, 3], "b": [3, 2, 1]}, "x", "y",
                                   log_y=True, invert_x=True, output_dir=out)
    assert _pdfs(out) == ["metric_vs_param.pdf"]
    ax = saved_figures[-1].axes[0]
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log") and ax.xaxis_inverted()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


def test_plot_nstar_vs_margin_with_fit(out, saved_figures):
    pub.plot_nstar_vs_margin([1, 2, 4], [100, 50, 25], fit_coeffs=(100, -1), tol=0.05,
                             output_dir=out)
    assert _pdfs(out) == ["nstar_vs_margin.pdf"]
    ax = saved_figures[-1].axes[0]
    fit = ax.get_lines()[1]
    np.testing.assert_allclose(fit.get_ydata(), 100 / fit.get_xdata())  # N* = a Δ^b
    assert "τ = 0.05" in ax.get_title()


def test_plot_recovery_heatmap(out, saved_figures):
    pub.plot_recovery_heatmap([[1.0, 0.5], [0.25, np.nan]], ["GMM", "label"], ["a", "b"],
                              fmt=".1f", good_is_high=False, output_dir=out)
    assert _pdfs(out) == ["recovery_heatmap.pdf"]
    ax = saved_figures[-1].axes[0]
    assert [t.get_text() for t in ax.texts] == ["1.0", "0.5", "0.2", "—"]


def test_plot_grouped_bars_with_ci(out, saved_figures):
    pub.plot_grouped_bars_with_ci(["a", "b"], [1.0, 2.0], [0.5, 1.8], [1.4, 2.5],
                                  baseline=1.0, output_dir=out)
    assert _pdfs(out) == ["grouped_bars_ci.pdf"]
    ax = saved_figures[-1].axes[0]
    bars = next(c for c in ax.containers if isinstance(c, ErrorbarContainer))
    # CI bounds are absolute values and become asymmetric error bars.
    segs = [s.tolist() for s in bars.lines[2][0].get_segments()]
    assert segs == [[[0.0, 0.5], [0.0, 1.4]], [[1.0, 1.8], [1.0, 2.5]]]


def test_publication_style_is_scoped_to_each_call(out, saved_figures):
    pub.plot_CE_vs_nprojections(N_PROJ, [0.3, 0.2, 0.1], output_dir=out)
    fig = saved_figures[-1]
    # 'classic' (80 dpi) + serif applied while drawing …
    assert fig.dpi == plt.style.library["classic"]["figure.dpi"]
    assert fig.axes[0].get_yticklabels()[0].get_fontfamily() == ["serif"]
    # … and not afterwards (rcParams are checked by the autouse fixture).
    assert plt.figure().dpi == mpl.rcParams["figure.dpi"]


def test_default_output_dir_is_created_on_first_save(tmp_path, monkeypatch):
    target = tmp_path / "default_out"
    monkeypatch.setattr(pub, "OUTPUT_DIR", target)
    pub.plot_DB_vs_nprojections(N_PROJ, [1, 2, 3])
    assert _pdfs(target) == ["G1_DB_vs_Nproj_clean.pdf"]


def test_every_figure_writes_only_into_output_dir(tmp_path, hist, out, monkeypatch):
    # Every public plot_* writes into out/ only.  PDF rendering is checked by the
    # tests above; here savefig just records its target, to keep this test fast.
    targets = []

    def record(self, fname, *args, **kwargs):
        targets.append(pathlib.Path(fname))
        pathlib.Path(fname).touch()

    monkeypatch.setattr(Figure, "savefig", record)
    figs = [n for n in vars(pub) if n.startswith("plot_")]
    calls = {
        "plot_phantom_label_map": ([np.zeros((2, 2), np.uint8), [nxs.AIR], 0.1], {}),
        "plot_bimodal_histogram": ([hist], {}),
        "plot_xray_marginal": ([hist], {}),
        "plot_neutron_marginal": ([hist], {}),
        "plot_reconstruction_slice": ([np.ones((2, 2))], {}),
        "plot_CE_vs_nprojections": ([N_PROJ, [1, 2, 3]], {}),
        "plot_DB_vs_nprojections": ([N_PROJ, [1, 2, 3]], {}),
        "plot_sigma_x_vs_nprojections": ([N_PROJ, {"A": [1, 2, 3]}], {}),
        "plot_sigma_n_vs_nprojections": ([N_PROJ, {"A": [1, 2, 3]}], {}),
        "plot_epsilon_vs_nprojections": ([N_PROJ, {"A": [1, 2, 3]}], {}),
        "plot_pairwise_overlap_vs_nprojections": ([N_PROJ, {("A", "B"): [1, 2, 3]}], {}),
        "plot_CE_all_geometries": ([{}, N_PROJ], {}),
        "plot_DB_all_geometries": ([{}, N_PROJ], {}),
        "plot_cross_sweep_heatmap": ([np.eye(2), [8, 16]], {}),
        "plot_xray_attenuation_spectra": ([[nxs.IRON], nxs.XRAY_E_KEV], {}),
        "plot_artifact_fingerprint": ([np.eye(2), ["a", "b"], ["c", "d"]], {}),
        "plot_metric_by_artifact": ([[1.0], ["a"]], {}),
        "plot_algorithm_cross_heatmap": ([np.eye(2), ["FBP", "SIRT"]], {}),
        "plot_rare_phase_failure": ([[0.1], [1.0], [2.0]], {}),
        "plot_metric_vs_param_multi": ([[1, 2], {"a": [1, 2]}, "x", "y"], {}),
        "plot_nstar_vs_margin": ([[1, 2], [10, 5]], {}),
        "plot_recovery_heatmap": ([np.eye(2), ["a", "b"], ["c", "d"]], {}),
        "plot_grouped_bars_with_ci": ([["a"], [1.0], [0.5], [1.5]], {}),
    }
    assert sorted(figs) == sorted(calls), "new publication function: add it here"
    for name, (args, kwargs) in calls.items():
        assert getattr(pub, name)(*args, output_dir=out, **kwargs) is None, name
    assert [p.name for p in tmp_path.iterdir()] == ["figures"]  # cwd untouched
    # No-op marginal wrappers and empty geometry sweeps save nothing.
    assert len(targets) == len(set(targets)) == 19
    assert all(t.parent == out and t.suffix == ".pdf" for t in targets)
    assert sorted(p.name for p in out.iterdir()) == sorted(t.name for t in targets)
    assert plt.get_fignums() == []


def test_legacy_diana_plots_names():
    import neutron_xray_sim.diana_plots as legacy

    assert legacy is pub
    for name in ("plot_CE_vs_nprojections", "plot_cross_sweep_heatmap", "MAT_COLORS"):
        assert getattr(nxs, name) is getattr(pub, name)
    # The documented top-level plot_bimodal_histogram is not the publication one.
    assert nxs.plot_bimodal_histogram is not pub.plot_bimodal_histogram


@pytest.mark.slow
def test_demo_writes_every_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "OUTPUT_DIR", tmp_path / "demo")
    pub._demo()
    assert len(_pdfs(tmp_path / "demo")) == 19


# ── Suspected bugs (strict xfail: they start passing once fixed) ──────────────

@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: the per-material / per-pair line plots zip the data with fixed-length "
    "marker (4) or colour (6) lists (publication.py:456-459, 508-511, 807-818, "
    "865-868, 919-923), so series beyond that are silently not drawn while their "
    "legend entries are also dropped"))
@pytest.mark.parametrize("func, n", [
    (pub.plot_sigma_x_vs_nprojections, 5),
    (pub.plot_sigma_n_vs_nprojections, 5),
    (pub.plot_epsilon_vs_nprojections, 5),
    (pub.plot_pairwise_overlap_vs_nprojections, 7),
])
def test_every_series_is_drawn(func, n, out, saved_figures):
    if func is pub.plot_pairwise_overlap_vs_nprojections:
        data = {(f"A{i}", f"B{i}"): [0.3, 0.2, 0.1] for i in range(n)}
    else:
        data = {f"M{i}": [0.3, 0.2, 0.1] for i in range(n)}
    func(N_PROJ, data, output_dir=out)
    assert len(saved_figures[-1].axes[0].get_lines()) == n


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: plot_xray_attenuation_spectra zips with 6 markers (publication.py:807, "
    "818); a phantom with more than 6 non-air materials (e.g. the 7-material "
    "jellyroll_battery preset) silently loses the extra curves"))
def test_attenuation_spectra_draw_every_material(out, saved_figures):
    ph = nxs.make_phantom("jellyroll_battery", N=16)
    pub.plot_xray_attenuation_spectra(ph.materials, nxs.XRAY_E_KEV, output_dir=out)
    assert len(saved_figures[-1].axes[0].get_lines()) == len(ph.materials) - 1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: with a shared extent, _draw_recon_panel passes the shared limits as "
    "the imshow extent (plotting/histograms.py:366-369), stretching the "
    "histogram image over [0, x_max] instead of its bin edges, so clusters are "
    "drawn at the wrong (μ_x, μ_n); affects plot_ground_truth_comparison, "
    "plot_artifact_survey and plot_cross_algorithm_grid"))
def test_reconstructed_panel_is_drawn_at_its_bin_edges(asymmetric_phantom, phantom_hist):
    fig = nxs.plot_ground_truth_comparison(asymmetric_phantom, phantom_hist)
    recon = next(a for a in fig.axes if a.get_title() == "Reconstructed")
    assert recon.get_images()[0].get_extent() == pytest.approx(phantom_hist.extent)
