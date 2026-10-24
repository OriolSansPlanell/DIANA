import matplotlib.pyplot as plt
import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.physics import neutron_spectra as ns

# λ[Å] = h / sqrt(2 m_n E)  ≈  9.045 / sqrt(E[meV])
LAMBDA_CONST = 9.045
KB_MEV_PER_K = 8.617333e-2
AVOGADRO = 6.02214076e23


def _flux_fraction_above(beam, e_cut):
    E, phi = beam.E_grid_meV, beam.phi
    return ns._trapz(np.where(E > e_cut, phi, 0.0), E) / ns._trapz(phi, E)


# ── Units and beam constructors ──────────────────────────────────────────────

def test_energy_to_wavelength_conversion():
    E = np.array([0.5, 2.0, 5.0, 25.3, 81.8])
    lam = ns._energy_meV_to_lambda_angstrom(E)
    # 9.045 is the textbook constant rounded to 4 digits → 1e-4 agreement.
    np.testing.assert_allclose(lam, LAMBDA_CONST / np.sqrt(E), rtol=1e-4)
    assert lam[3] == pytest.approx(1.798, abs=1e-3)   # 2200 m/s thermal neutrons
    assert lam[4] == pytest.approx(1.000, abs=1e-3)   # 81.8 meV ↔ 1 Å


def test_mono_beams():
    th = nxs.thermal_beam()
    assert th.is_mono and th.E_mean_meV == pytest.approx(25.3)
    assert th.lambda_mean_angstrom == pytest.approx(1.798, abs=1e-3)
    cold = nxs.cold_mono_beam(5.0)
    assert cold.is_mono and cold.name == "cold_mono_5.0meV"
    np.testing.assert_array_equal(cold.phi, [1.0])
    assert cold.lambda_mean_angstrom == pytest.approx(LAMBDA_CONST / np.sqrt(5.0), rel=1e-4)
    assert "4.04 Å" in cold.description and "mono" in repr(cold)
    # Colder beam → longer wavelength.
    assert nxs.NEUTRON_MODES["cold_2meV"].lambda_mean_angstrom > cold.lambda_mean_angstrom


@pytest.mark.parametrize("E", [0.0, -1.0])
def test_cold_mono_beam_rejects_nonpositive_energy(E):
    with pytest.raises(ValueError, match="E_meV must be > 0"):
        nxs.cold_mono_beam(E)


def test_neutron_beam_validates_and_renormalises():
    E = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same shape"):
        nxs.NeutronBeam("bad", E, np.ones(2))
    with pytest.raises(ValueError, match="positive"):
        nxs.NeutronBeam("bad", E, np.zeros(3))
    b = nxs.NeutronBeam("ok", E, np.array([2.0, 4.0, 2.0]))
    assert b.phi.sum() == pytest.approx(1.0)
    assert b.E_mean_meV == pytest.approx(2.0)          # symmetric spectrum
    assert "poly" in repr(b)


def test_maxwellian_moments():
    # Pure Maxwellian φ(E) ∝ E·exp(-E/kT): peak at kT, <E> = 2kT and
    # <λ> = C·<E^-1/2> = C·Γ(3/2)/sqrt(kT).  Wide, fine grid so truncation
    # and quadrature errors are < 1e-3.
    T = 35.0
    kT = KB_MEV_PER_K * T
    b = nxs.cold_poly_beam(T_K=T, epsilon=0.0, n_bins=4000, E_min_meV=0.01, E_max_meV=60.0)
    assert not b.is_mono and b.name == "cold_poly_T35K"
    assert "T=35 K" in b.description
    assert b.phi.sum() == pytest.approx(1.0) and np.all(b.phi > 0)
    dE = b.E_grid_meV[1] - b.E_grid_meV[0]
    assert abs(b.E_grid_meV[np.argmax(b.phi)] - kT) <= dE
    assert b.E_mean_meV == pytest.approx(2 * kT, rel=1e-3)
    assert b.lambda_mean_angstrom == pytest.approx(
        LAMBDA_CONST * 0.5 * np.sqrt(np.pi) / np.sqrt(kT), rel=2e-3)


def test_colder_moderator_gives_longer_wavelengths():
    beams = [nxs.cold_poly_beam(T_K=T) for T in (20.0, 35.0, 100.0)]
    lam = [b.lambda_mean_angstrom for b in beams]
    E = [b.E_mean_meV for b in beams]
    assert lam[0] > lam[1] > lam[2]
    assert E[0] < E[1] < E[2]


def test_thermal_tail_hardens_spectrum():
    # The ε/E term adds flux at high energy: more of the beam above 15 meV.
    fr = [_flux_fraction_above(nxs.cold_poly_beam(epsilon=eps), 15.0)
          for eps in (0.0, 0.08, 0.3)]
    assert fr[0] < fr[1] < fr[2]


def test_ill_next_beam():
    b = nxs.ill_next_beam()
    ref = nxs.cold_poly_beam(T_K=35.0, epsilon=0.08)
    np.testing.assert_allclose(b.phi, ref.phi)
    assert b.name == "ILL-NeXT" and "Tengattini" in b.description
    # Cold-source peak at kT(35 K) ≈ 3.0 meV, <λ> in the cold range.
    dE = b.E_grid_meV[1] - b.E_grid_meV[0]
    assert abs(b.E_grid_meV[np.argmax(b.phi)] - KB_MEV_PER_K * 35.0) <= dE
    assert 4.0 < b.lambda_mean_angstrom < 5.0
    assert set(nxs.NEUTRON_MODES) == {"thermal", "cold_5meV", "cold_2meV", "ill_next"}


# ── Cross sections ───────────────────────────────────────────────────────────

def test_mu_n_at_thermal_energy_matches_tables():
    # Iron, Sears (1992) bound-atom values: σ_tot = 2.56 + 11.22 + 0.40 b.
    expected_fe = 7.874 * AVOGADRO / 55.845 * 14.18e-24
    assert ns.mu_n_at_energy({"Fe": 1}, 7.874, 25.3) == pytest.approx(expected_fe, rel=1e-3)
    # Same formula through the Material builder → identical (shared tables).
    water = nxs.material_from_formula("w", "w", "H2O", density_gcc=1.0)
    mu_w = ns.mu_n_at_energy({"H": 2, "O": 1}, 1.0, 25.3)
    assert mu_w == pytest.approx(water.mu_n, rel=1e-12)
    assert mu_w == pytest.approx(5.65, rel=0.02)


def test_absorption_follows_one_over_v():
    # Indium is absorption-dominated.  Only σ_abs scales as sqrt(E0/E), so
    # Σ(E0/4) − Σ(E0) = Σ_abs(E0)·(2−1) and Σ(E0/16) − Σ(E0/4) = Σ_abs(E0)·(4−2).
    E0 = 25.3
    mu = {E: ns.mu_n_at_energy({"In": 1}, 7.31, E) for E in (E0, E0 / 4, E0 / 16)}
    d1 = mu[E0 / 4] - mu[E0]
    d2 = mu[E0 / 16] - mu[E0 / 4]
    assert d2 / d1 == pytest.approx(2.0, rel=1e-10)
    # Oxygen is a near-pure coherent scatterer: Σ barely changes 25 → 2 meV.
    mu_o = [ns.mu_n_at_energy({"O": 1}, 1.0, E) for E in (25.3, 2.0)]
    assert mu_o[1] / mu_o[0] - 1 < 1e-3


def test_mu_n_unknown_element_raises():
    with pytest.raises(KeyError):
        ns.mu_n_at_energy({"Xx": 1}, 1.0, 25.3)


@pytest.mark.xfail(strict=True, reason=(
    "ATOMIC_MASS lookup (neutron_spectra.py:347) raises a bare KeyError before "
    "the actionable NEUTRON_XS message at :353 is reached"))
def test_mu_n_unknown_element_message_is_actionable():
    with pytest.raises(KeyError, match="NEUTRON_XS"):
        ns.mu_n_at_energy({"Xx": 1}, 1.0, 25.3)


@pytest.mark.parametrize("E", [2.0, 5.0, 25.3])
def test_mu_n_for_mono_beam_equals_mu_at_energy(E):
    f = {"H": 2, "O": 1}
    assert ns.mu_n_for_beam(f, 1.0, nxs.cold_mono_beam(E)) == ns.mu_n_at_energy(f, 1.0, E)


def test_mu_n_for_poly_beam_is_flux_weighted():
    beam = nxs.ill_next_beam()
    f, rho = {"In": 1}, 7.31
    mu_grid = np.array([ns.mu_n_at_energy(f, rho, E) for E in beam.E_grid_meV])
    eff = ns.mu_n_for_beam(f, rho, beam)
    assert mu_grid.min() < eff < mu_grid.max()
    # 1/sqrt(E) is convex → spectrum average exceeds the value at <E> (Jensen).
    assert eff > ns.mu_n_at_energy(f, rho, beam.E_mean_meV)


# ── Look-up tables ───────────────────────────────────────────────────────────

def test_lut_thermal_uses_stored_values():
    mats = list(nxs.MATERIALS.values())
    lut = nxs.mu_n_lut_for_beam(mats, nxs.thermal_beam())
    assert lut.dtype == np.float32
    np.testing.assert_allclose(lut, [m.mu_n for m in mats], rtol=1e-6)


def test_lut_cold_mono_scales_absorption_only():
    E = 5.0
    lut = nxs.mu_n_lut_for_beam([nxs.WATER, nxs.IRON], nxs.cold_mono_beam(E))
    expected = [m.mu_n_abs * np.sqrt(25.3 / E) + m.mu_n_coh + m.mu_n_inc
                for m in (nxs.WATER, nxs.IRON)]
    np.testing.assert_allclose(lut, expected, rtol=1e-6)


def test_lut_poly_matches_formula_path():
    # Material-based LUT and formula-based Σ_eff are two independent code paths.
    beam = nxs.ill_next_beam()
    water = nxs.material_from_formula("w", "w", "H2O", 1.0)
    lif = nxs.material_from_formula("LiF", "LiF", "LiF", 2.635)
    lut = nxs.mu_n_lut_for_beam([water, lif], beam)
    np.testing.assert_allclose(
        lut,
        [ns.mu_n_for_beam({"H": 2, "O": 1}, 1.0, beam),
         ns.mu_n_for_beam({"Li": 1, "F": 1}, 2.635, beam)],
        rtol=1e-5)  # float32 LUT


def test_lut_label_to_mat():
    lut = nxs.mu_n_lut_for_beam(None, nxs.thermal_beam(),
                                label_to_mat={0: nxs.AIR, 3: nxs.WATER})
    assert lut.shape == (4,)
    np.testing.assert_allclose(lut, [0.0, 0.0, 0.0, nxs.WATER.mu_n], rtol=1e-6)


def _lif():
    return nxs.material_from_formula("LiF", "LiF", "LiF", 2.635)


def test_spectrum_lut_thin_sample_limit():
    beam = nxs.ill_next_beam()
    mats = [nxs.material_from_formula("w", "w", "H2O", 1.0), _lif()]
    E, w, mu = nxs.mu_n_spectrum_lut(mats, beam)
    assert E.shape == w.shape == (beam.E_grid_meV.size,)
    assert mu.shape == (2, E.size) and mu.dtype == np.float32
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    # Σ_k w_k μ(E_k) reproduces the flux-weighted (thin-sample) LUT.
    np.testing.assert_allclose(mu @ w, nxs.mu_n_lut_for_beam(mats, beam), rtol=1e-5)
    # Monochromatic: one bin, weight 1, μ equal to the mono LUT.
    E1, w1, mu1 = nxs.mu_n_spectrum_lut(mats, nxs.cold_mono_beam(5.0))
    np.testing.assert_array_equal(w1, [1.0])
    np.testing.assert_allclose(mu1[:, 0], nxs.mu_n_lut_for_beam(mats, nxs.cold_mono_beam(5.0)))


def test_spectrum_lut_rebinning_converges():
    # Each group is evaluated at its flux-weighted mean energy; since
    # 1/sqrt(E) is convex this underestimates Σ_eff (Jensen) and the error
    # shrinks as the groups get narrower.
    beam = nxs.ill_next_beam()
    lif = _lif()
    thin = nxs.mu_n_lut_for_beam([lif], beam)[0]
    errs = []
    for n_bins in (10, 20, 50):
        E, w, mu = nxs.mu_n_spectrum_lut([lif], beam, n_bins=n_bins)
        assert E.size == n_bins and np.all(np.diff(E) > 0)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        errs.append(float(mu[0] @ w) / thin - 1)
    assert errs[0] < errs[1] < errs[2] <= 0
    assert abs(errs[2]) < 5e-3


def test_self_shielding_lowers_apparent_mu():
    # Neutron "beam hardening": a 1/v absorber (⁶Li in LiF) removes the coldest
    # neutrons first, so the apparent Σ = -ln T / L falls with thickness and
    # tends to the flux-weighted value as L → 0.
    beam = nxs.ill_next_beam()
    lif = _lif()
    E, w, mu = nxs.mu_n_spectrum_lut([lif], beam)
    thin = nxs.mu_n_lut_for_beam([lif], beam)[0]
    w, mu = w.astype(float), mu[0].astype(float)

    def apparent(L):
        return -np.log(w @ np.exp(-mu * L)) / L

    L = [1e-4, 0.05, 0.2, 0.5]
    app = [apparent(x) for x in L]
    assert app[0] == pytest.approx(thin, rel=1e-3)
    assert all(a > b for a, b in zip(app, app[1:]))
    assert app[-1] < 0.85 * thin


def test_plot_spectra_default_catalogue():
    fig = nxs.plot_spectra()
    ax_e, ax_l = fig.axes
    assert len(ax_e.get_legend().get_texts()) == len(nxs.NEUTRON_MODES)
    # Mono beams are drawn as vertical lines at their energy / wavelength.
    x_e = {round(float(ln.get_xdata()[0]), 3) for ln in ax_e.get_lines()
           if np.ptp(ln.get_xdata()) == 0}
    assert {25.3, 5.0, 2.0} <= x_e
    x_l = {round(float(ln.get_xdata()[0]), 2) for ln in ax_l.get_lines()
           if np.ptp(ln.get_xdata()) == 0}
    assert round(nxs.NEUTRON_MODES["cold_5meV"].lambda_mean_angstrom, 2) in x_l
    plt.close("all")


@pytest.mark.xfail(strict=True, reason=(
    "plot_spectra plots |dφ/dλ| instead of φ(λ) = φ(E)·|dE/dλ| "
    "(neutron_spectra.py:600)"))
def test_plot_spectra_wavelength_curve_is_jacobian_transform():
    # For a Maxwellian φ(E) ∝ E·exp(-E/kT), φ(λ) ∝ λ⁻⁵·exp(-C²/(λ²kT)) has a
    # single maximum at λ = C·sqrt(2/(5kT)) ≈ 3.29 Å for T = 35 K.
    T = 35.0
    beam = nxs.cold_poly_beam(T_K=T, epsilon=0.0)
    fig = nxs.plot_spectra({"mb": beam})
    lam, y = fig.axes[1].get_lines()[0].get_data()
    plt.close("all")
    n_max = np.sum((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))
    lam_peak = LAMBDA_CONST * np.sqrt(2.0 / (5.0 * KB_MEV_PER_K * T))
    assert n_max == 1
    assert lam[np.argmax(y)] == pytest.approx(lam_peak, abs=0.05)
