"""Fourier fusion of tomography and laminography volumes."""
import numpy as np
import pytest

from neutron_xray_sim.reconstruction.fusion import (
    FusionConfig,
    fuse_volumes_fourier,
    fusion_weight_tomography,
)

CFG = FusionConfig(voxel_um_zyx=(1.0, 1.0, 1.0), tomography_native_voxel_um=2.0,
                   missing_cone_half_angle_deg=45.0, angular_transition_deg=4.0,
                   radial_transition_fraction=0.1)
SHAPE = (16, 16, 16)


def test_weight_shape_range_and_dc():
    W = fusion_weight_tomography(SHAPE, CFG)
    assert W.shape == SHAPE and W.dtype == np.float32
    assert W.min() >= 0.0 and W.max() <= 1.0
    assert W[0, 0, 0] == 1.0
    lam_dc = FusionConfig(**{**CFG.__dict__, "preserve_dc_from": "laminography"})
    assert fusion_weight_tomography(SHAPE, lam_dc)[0, 0, 0] == 0.0


def test_weight_takes_tomography_only_inside_the_missing_cone():
    # Frequencies are fftfreq(16, d=1 µm) = k/16 µm⁻¹; tomography Nyquist is 0.25.
    W = fusion_weight_tomography(SHAPE, CFG)
    assert W[1, 0, 0] == pytest.approx(1.0)       # along the laminography axis, low k
    assert W[0, 1, 0] == pytest.approx(0.0)       # perpendicular to it
    assert W[0, 0, 1] == pytest.approx(0.0)
    assert W[6, 0, 0] == pytest.approx(0.0)       # 0.375 µm⁻¹: above tomo Nyquist
    assert W[2, 1, 0] == pytest.approx(1.0)       # 26.6° from the axis, inside 45°
    assert W[1, 2, 0] == pytest.approx(0.0)       # 63.4° from the axis, outside


def test_weight_is_symmetric_in_k():
    # Real volumes need W(k) = W(-k) so the fused volume stays real.
    W = fusion_weight_tomography(SHAPE, CFG)
    W_neg = np.roll(W[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
    np.testing.assert_allclose(W, W_neg, atol=1e-6)


def test_fusing_a_volume_with_itself_is_the_identity(rng):
    vol = rng.random(SHAPE).astype(np.float32)
    fused, _ = fuse_volumes_fourier(vol, vol, CFG)
    np.testing.assert_allclose(fused, vol, atol=1e-5)


def test_fusion_takes_low_axial_frequencies_from_tomography(rng):
    z = np.arange(SHAPE[0])[:, None, None]
    x = np.arange(SHAPE[2])[None, None, :]
    axial = np.cos(2 * np.pi * z / 16) * np.ones(SHAPE)      # k along the axis
    lateral = np.cos(2 * np.pi * x / 16) * np.ones(SHAPE)    # k perpendicular
    tomo = axial + 0.5
    lamino = lateral + 0.5 + 0.1 * rng.standard_normal(SHAPE)
    fused, W = fuse_volumes_fourier(lamino, tomo, CFG)
    F = np.fft.fftn(fused) / fused.size
    assert F[1, 0, 0].real == pytest.approx(0.5, abs=1e-6)    # axial wave from tomo
    assert abs(F[0, 0, 1].real - 0.5) < 0.02                  # lateral wave from lamino
    assert F[0, 0, 0].real == pytest.approx(0.5, abs=1e-6)    # DC from tomography
    np.testing.assert_array_equal(W, fusion_weight_tomography(SHAPE, CFG))


def test_fusion_requires_matching_grids():
    with pytest.raises(ValueError, match="share shape"):
        fuse_volumes_fourier(np.zeros((4, 4, 4)), np.zeros((4, 4, 5)), CFG)
