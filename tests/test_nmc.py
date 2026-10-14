"""NMC particle-slab phantoms, density sweeps and particle metrics."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from neutron_xray_sim.analysis.fusion_metrics import (
    match_reconstruction_to_truth,
    normalized_rmse,
    particle_shape_table,
    segment_particles,
)
from neutron_xray_sim.phantoms import density_sweep as ds
from neutron_xray_sim.phantoms.nmc import (
    NMCParticleConfig,
    generate_nmc_particle_slab,
    make_nmc_material,
)

SMALL = NMCParticleConfig(shape_zyx=(24, 48, 48), voxel_um_zyx=(1.0, 1.0, 1.0),
                          n_particles=12, diameter_mean_um=9.0, diameter_sigma_um=1.0,
                          diameter_min_um=6.0, diameter_max_um=12.0,
                          slab_thickness_um=20.0, seed=3)


@pytest.fixture(scope="module")
def slab():
    return generate_nmc_particle_slab(SMALL)


# ── phantom generation ───────────────────────────────────────────────────────

def test_slab_phantom_shape_and_materials(slab):
    phantom, truth = slab
    assert phantom.shape == SMALL.shape_zyx
    assert [m.name for m in phantom.materials][1] == "NMC811"
    assert set(np.unique(phantom.label_vol)) <= {0, 1}
    assert phantom.voxel_cm == pytest.approx(1e-4)
    assert truth.attrs["requested_particles"] == 12
    assert truth.attrs["accepted_particles"] == len(truth) > 0


def test_particles_do_not_overlap_and_stay_in_the_slab(slab):
    _, truth = slab
    c = truth[["center_z_um", "center_y_um", "center_x_um"]].to_numpy()
    r = truth["radius_um"].to_numpy()
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    gap = d - (r[:, None] + r[None]) - SMALL.packing_margin_um
    assert (gap[~np.eye(len(r), dtype=bool)] >= 0).all()
    assert (np.abs(c[:, 0]) + r <= SMALL.slab_thickness_um / 2 + 1e-9).all()
    assert (np.abs(c[:, 1:]) + r[:, None] <= 24.0 + 1e-9).all()
    assert truth["diameter_um"].between(6.0, 12.0).all()


def test_voxelised_volume_matches_the_truth_table(slab):
    phantom, truth = slab
    voxels = np.count_nonzero(phantom.label_vol)        # 1 µm³ voxels
    assert voxels == pytest.approx(truth["volume_um3"].sum(), rel=0.05)


def test_generation_is_seeded(slab):
    _, truth = slab
    _, again = generate_nmc_particle_slab(SMALL)
    pd.testing.assert_frame_equal(truth, again)
    _, other = generate_nmc_particle_slab(replace(SMALL, seed=4))
    assert not truth.equals(other)


def test_lognormal_diameters_have_the_requested_moments():
    from neutron_xray_sim.phantoms.nmc import _sample_diameters
    cfg = replace(SMALL, distribution="lognormal", n_particles=20000,
                  diameter_min_um=0.0, diameter_max_um=1e3)
    d = _sample_diameters(cfg, np.random.default_rng(0))
    assert d.mean() == pytest.approx(9.0, rel=0.01)
    assert d.std() == pytest.approx(1.0, rel=0.03)


def test_bad_diameter_settings_are_rejected():
    with pytest.raises(ValueError, match="distribution"):
        generate_nmc_particle_slab(replace(SMALL, distribution="uniform"))
    with pytest.raises(RuntimeError, match="truncation"):
        generate_nmc_particle_slab(replace(SMALL, diameter_min_um=30.0,
                                           diameter_max_um=31.0))


def test_anisotropic_voxels_use_the_geometric_mean():
    cfg = replace(SMALL, voxel_um_zyx=(2.0, 1.0, 1.0), n_particles=2)
    phantom, _ = generate_nmc_particle_slab(cfg)
    assert phantom.voxel_cm == pytest.approx(2 ** (1 / 3) * 1e-4)


def test_nmc_material_attenuates_both_modalities():
    mat = make_nmc_material(SMALL)
    assert mat.density_gcc == pytest.approx(4.8)
    assert mat.mu_n > 0 and mat.mu_x_at(100.0) > 0


# ── density sweep ────────────────────────────────────────────────────────────

def test_expected_sphere_volume_is_the_gaussian_third_moment(rng):
    assert ds.expected_sphere_volume_um3(10.0, 0.0) == pytest.approx(np.pi / 6 * 1e3)
    d = rng.normal(20.0, 4.0, 400_000)
    assert ds.expected_sphere_volume_um3(20.0, 4.0) == pytest.approx(
        np.mean(np.pi / 6 * d ** 3), rel=0.01)


def test_particle_count_estimate():
    region = ds.sample_region_volume_um3(SMALL)
    assert region == pytest.approx(20.0 * 48 * 48)
    n1 = ds.estimate_particle_count_for_fraction(SMALL, 0.1)
    n2 = ds.estimate_particle_count_for_fraction(SMALL, 0.2)
    assert n2 == pytest.approx(2 * n1, abs=1)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="between zero and one"):
            ds.estimate_particle_count_for_fraction(SMALL, bad)


def test_build_density_config():
    cfg = ds.build_density_config(SMALL, 0.1, seed=7, particle_count_scale=1.0)
    assert cfg.seed == 7
    assert cfg.n_particles == ds.estimate_particle_count_for_fraction(SMALL, 0.1)
    assert cfg.shape_zyx == SMALL.shape_zyx


def test_solid_fractions_agree(slab):
    phantom, truth = slab
    analytical = ds.analytical_solid_fraction(truth, SMALL)
    voxelised = ds.voxelized_solid_fraction(phantom.label_vol, SMALL)
    assert 0 < analytical < 1
    assert voxelised == pytest.approx(analytical, rel=0.05)
    assert ds.analytical_solid_fraction(truth.iloc[0:0], SMALL) == 0.0


def test_density_preflight_writes_every_case(tmp_path):
    base = replace(SMALL, n_particles=1)
    summary, phantoms = ds.phantom_density_preflight(base, [0.05, 0.1], tmp_path,
                                                     seeds=[1, 2])
    assert list(summary["case"]) == ["phi_0p050", "phi_0p100"]
    assert (summary["accepted_particles"] <= summary["requested_particles"]).all()
    assert (tmp_path / "density_preflight_summary.csv").exists()
    for case in summary["case"]:
        for name in ("particle_truth.csv", "ground_truth_labels.npy",
                     "phantom_config.json"):
            assert (tmp_path / case / name).exists()
    assert set(phantoms) == {0.05, 0.1}
    with pytest.raises(ValueError, match="same length"):
        ds.phantom_density_preflight(base, [0.05, 0.1], tmp_path, seeds=[1])


# ── particle metrics ─────────────────────────────────────────────────────────

def test_normalized_rmse():
    ref = np.array([1.0, 2.0, 3.0])
    assert normalized_rmse(ref, ref) == 0.0
    assert normalized_rmse(ref, 2 * ref) == pytest.approx(1.0)


def test_shape_table_of_a_single_sphere():
    n, r = 32, 8.0
    z, y, x = np.mgrid[:n, :n, :n] + 0.5 - n / 2
    labels = ((z ** 2 + y ** 2 + x ** 2) <= r ** 2).astype(np.int32)
    table = particle_shape_table(labels, (1.0, 1.0, 1.0))
    assert len(table) == 1
    row = table.iloc[0]
    assert row["equivalent_diameter_um"] == pytest.approx(2 * r, rel=0.02)
    assert row["major_to_minor_ratio"] == pytest.approx(1.0, abs=0.02)
    assert 0.85 < row["sphericity"] <= 1.0          # voxelised surface is rough
    for c in ("centroid_z_um", "centroid_y_um", "centroid_x_um"):
        assert row[c] == pytest.approx(0.0, abs=1e-9)


def test_segmentation_recovers_the_particles(slab, rng):
    phantom, truth = slab
    vol = phantom.label_vol.astype(np.float32) + 0.05 * rng.standard_normal(phantom.shape)
    labels = segment_particles(vol, (1.0, 1.0, 1.0), smooth_sigma_um=0.7,
                               seed_min_distance_um=3.0, min_equivalent_diameter_um=4.0,
                               remove_border=False)
    table = particle_shape_table(labels, (1.0, 1.0, 1.0))
    matched = match_reconstruction_to_truth(table, truth, max_distance_um=2.0)
    assert len(matched) >= 0.8 * len(truth)
    assert matched["match_distance_um"].max() <= 2.0
    assert matched["diameter_relative_error"].abs().median() < 0.1
    assert matched["truth_particle_id"].is_unique


def test_segmentation_rejects_empty_volumes():
    with pytest.raises(ValueError, match="no finite"):
        segment_particles(np.full((4, 4, 4), np.nan))


def test_matching_handles_empty_tables(slab):
    _, truth = slab
    empty = particle_shape_table(np.zeros((4, 4, 4), np.int32), (1.0, 1.0, 1.0))
    assert match_reconstruction_to_truth(empty, truth).empty
