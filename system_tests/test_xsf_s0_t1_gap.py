"""S0-T1 gap and Eq. (15) for XSF-TDA, without changing default eigenvalues."""

from pathlib import Path
import sys

import numpy as np
import pytest
from pyscf import dft, gto, scf

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.utils.backend import set_backend


ATOM = (
    ("C", (0.00, 0.00, 0.00)),
    ("H", (0.00, 0.95, 0.15)),
    ("H", (0.10, -0.75, 0.65)),
)


def _ch2(basis="sto-3g"):
    return gto.M(atom=ATOM, basis=basis, spin=2, unit="Angstrom", verbose=0)


@pytest.fixture(scope="module")
def ch2_rohf():
    set_backend("cpu")
    mf = scf.ROHF(_ch2()).run(conv_tol=1e-11)
    assert mf.converged
    return mf


@pytest.fixture(scope="module")
def ch2_pbe():
    set_backend("cpu")
    mf = dft.ROKS(_ch2())
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mf


def _oo_slice(td):
    dim3 = td.nc * td.nv + td.nc * td.no + td.no * td.nv
    return dim3


def test_eq12_vector_is_the_removed_direction():
    """The Eq. (12) amplitude is normalized and orthogonal to get_vect."""
    for n_open in (2, 3, 4):
        holder = type("Holder", (), {"no": n_open})()
        vects = XSF_TDA_down.get_vect(holder)
        amplitude = np.zeros(n_open * n_open)
        scale = 1.0 / np.sqrt(n_open)
        for orbital in range(n_open):
            amplitude[orbital * n_open + orbital] = scale
        assert vects.shape == (n_open * n_open, n_open * n_open - 1)
        np.testing.assert_allclose(vects.T @ vects, np.eye(vects.shape[1]), atol=1e-12)
        np.testing.assert_allclose(vects.T @ amplitude, 0.0, atol=1e-12)
        assert abs(np.linalg.norm(amplitude) - 1.0) < 1e-12


def test_rohf_omega0_is_zero_and_matches_response(ch2_rohf):
    """Pure HF satisfies Eq. (13): omega0 = x^T [A-] x = 0, and the matvec matches A^-."""
    set_backend("cpu")
    td = XSF_TDA_down(ch2_rohf, method=2, SA=3, davidson=False)
    energies, _vectors = td.kernel(nstates=6)
    energies = np.asarray(energies)
    saved_e = np.array(td.e)
    saved_v = np.array(td.v)
    assert td.re
    assert td.A.shape[0] == (td.nc + td.no) * (td.no + td.nv) - 1

    amplitude = td.intra_multiplet_vector()
    assert abs(np.linalg.norm(amplitude) - 1.0) < 1e-12
    dim3 = _oo_slice(td)
    np.testing.assert_allclose(td.get_vect().T @ amplitude[dim3:], 0.0, atol=1e-12)
    ax = td._apply_A_minus(amplitude)
    matrix_omega0 = float(amplitude @ np.asarray(td.sf_tda_A) @ amplitude)
    response_omega0 = float(amplitude @ ax)
    assert abs(matrix_omega0) < 1e-8
    assert abs(response_omega0 - matrix_omega0) < 1e-8

    rng = np.random.default_rng(12)
    probe = rng.normal(size=amplitude.size)
    probe /= np.linalg.norm(probe)
    np.testing.assert_allclose(
        td._apply_A_minus(probe),
        np.asarray(td.sf_tda_A) @ probe,
        atol=1e-8,
    )

    report = td.s0_t1_report(print_report=False)
    np.testing.assert_allclose(td.e, saved_e)
    np.testing.assert_allclose(td.v, saved_v)
    assert report.reference_is_triplet
    assert report.eq15_applied is False
    assert report.quoted_zero == "ROKS"
    assert report.omega0_includes_delta_a is False
    assert report.a_minus_source == "sf_tda_A"
    assert report.t1.kind == "eq12"
    assert report.t1.in_spectrum is False
    assert report.s0.root == 0
    assert report.s0.spectroscopic == "S0"
    assert max(row.overlap for row in report.roots) < 1e-8
    assert report.gap_ha == pytest.approx(report.s0_vs_roks_ha - report.omega0_ha)
    assert report.gap_ev == pytest.approx(energies[0] - report.omega0_ev)
    assert report.e_t1_ha == pytest.approx(report.omega0_ha)
    assert report.e_s0_ha == pytest.approx(report.s0_vs_roks_ha)
    assert len(report.triplets) == 1

    forced = td.s0_t1_report(eq15=True, print_report=False)
    assert forced.eq15_applied is True
    assert forced.quoted_zero == "T1"
    assert forced.e_t1_ha == pytest.approx(0.0)
    assert forced.e_s0_ha == pytest.approx(forced.gap_ha)
    assert forced.gap_ha == pytest.approx(report.gap_ha)
    np.testing.assert_allclose(td.e, saved_e)


def test_sa0_keeps_t1_out_of_the_singlet_list(ch2_rohf):
    """With the intra-multiplet vector left in the space, it is T1 and not S0."""
    set_backend("cpu")
    td = XSF_TDA_down(ch2_rohf, method=2, SA=0, davidson=False)
    td.kernel(nstates=6)
    assert not td.re
    report = td.s0_t1_report(print_report=False)
    assert report.t1.in_spectrum is True
    assert report.t1.overlap == pytest.approx(1.0, abs=5e-3)
    assert report.s0.root != report.t1.root
    assert report.s0.kind == "target-spin"
    assert abs(report.omega0_ha) < 1e-8
    # Eq. (12) defines E(T1) by the expectation value. A^- does not have to
    # have that vector as an eigenvector, so the nearby root can sit elsewhere.
    assert report.t1.vs_roks_ha == pytest.approx(report.omega0_ha)
    if abs(report.t1.eigenvalue_ha - report.omega0_ha) > 1e-5:
        assert "expectation" in report.note or "omega0" in report.note
    singlet_roots = {row.root for row in report.singlets}
    assert report.t1.root not in singlet_roots
    assert report.gap_ha == pytest.approx(report.s0_vs_roks_ha - report.omega0_ha)


def test_default_kernel_energies_are_unchanged_by_the_report(ch2_rohf):
    set_backend("cpu")
    plain = XSF_TDA_down(ch2_rohf, method=2, SA=3, davidson=False)
    energies, vectors = plain.kernel(nstates=4)
    assert not hasattr(plain, "s0_t1")

    reported = XSF_TDA_down(ch2_rohf, method=2, SA=3, davidson=False)
    energies_reported, vectors_reported = reported.kernel(nstates=4, s0_t1=True)
    np.testing.assert_allclose(energies_reported, energies, atol=0.0)
    actual = np.asarray(vectors_reported)
    desired = np.asarray(vectors)
    for column in range(actual.shape[1]):
        overlap = abs(np.vdot(actual[:, column], desired[:, column]))
        assert overlap == pytest.approx(1.0, abs=1e-10)
    assert reported.s0_t1.gap_ev == pytest.approx(energies[0] - reported.s0_t1.omega0_ev)
    delta_s2 = reported.analyse(s0_t1=False)
    assert delta_s2.shape == (4,)


def test_davidson_omega0_matches_dense(ch2_rohf):
    set_backend("cpu")
    dense = XSF_TDA_down(ch2_rohf, method=2, SA=3, davidson=False)
    dense.kernel(nstates=4)
    davidson = XSF_TDA_down(
        ch2_rohf, method=2, SA=3, davidson=True, davidson_backend="cpu",
    )
    davidson.kernel(nstates=4)
    assert not hasattr(davidson, "sf_tda_A")
    report = davidson.s0_t1_report(print_report=False)
    assert report.a_minus_source == "response"
    assert report.omega0_ha == pytest.approx(dense.omega0(), abs=1e-8)
    np.testing.assert_allclose(davidson.e, dense.e, atol=2e-7)
    assert report.gap_ha == pytest.approx(
        dense.s0_t1_report(print_report=False).gap_ha, abs=2e-7,
    )


def test_alda0_applies_eq15_without_shifting_stored_eigenvalues(ch2_pbe):
    set_backend("cpu")
    alda = XSF_TDA_down(ch2_pbe, method=0, SA=3, davidson=False)
    alda.kernel(nstates=4)
    saved = np.array(alda.e)
    amplitude = alda.intra_multiplet_vector()
    probe = np.ones(amplitude.size)
    probe /= np.linalg.norm(probe)
    np.testing.assert_allclose(
        alda._apply_A_minus(probe),
        np.asarray(alda.sf_tda_A) @ probe,
        atol=1e-7,
    )
    alda_report = alda.s0_t1_report(print_report=False)
    np.testing.assert_allclose(alda.e, saved)
    assert alda_report.eq15_applied is True
    assert alda_report.quoted_zero == "T1"
    assert alda_report.e_t1_ha == pytest.approx(0.0)
    assert alda_report.e_s0_ha == pytest.approx(alda_report.gap_ha)
    assert alda_report.gap_ha == pytest.approx(
        alda_report.s0_vs_roks_ha - alda_report.omega0_ha,
    )
    np.testing.assert_allclose(
        alda_report.roots_eq15_ha,
        saved[:alda.nstates] - alda_report.omega0_ha,
    )

    disabled = alda.s0_t1_report(eq15=False, print_report=False)
    assert disabled.eq15_applied is False
    assert disabled.quoted_zero == "ROKS"
    assert disabled.gap_ha == pytest.approx(alda_report.gap_ha)
    assert disabled.e_t1_ha == pytest.approx(disabled.omega0_ha)

    collinear = XSF_TDA_down(ch2_pbe, method=2, SA=3, davidson=False)
    collinear.kernel(nstates=4)
    collinear_report = collinear.s0_t1_report(print_report=False)
    assert collinear_report.eq15_applied is False
    assert collinear_report.quoted_zero == "ROKS"
    # ALDA0 drops GGA gradients in the spin-flip kernel, so omega0 moves.
    assert abs(alda_report.omega0_ha - collinear_report.omega0_ha) > 1e-6


def test_mcol_response_matches_dense_a_minus(ch2_pbe):
    """Davidson's multicollinear response is the same A^- as the dense matrix."""
    set_backend("cpu")
    td = XSF_TDA_down(
        ch2_pbe, method=1, SA=3, davidson=False, collinear_samples=8,
    )
    td.kernel(nstates=2)
    amplitude = td.intra_multiplet_vector()
    probe = np.ones(amplitude.size)
    probe /= np.linalg.norm(probe)
    np.testing.assert_allclose(
        td._apply_A_minus(probe),
        np.asarray(td.sf_tda_A) @ probe,
        atol=1e-6,
    )
    report = td.s0_t1_report(print_report=False)
    assert report.eq15_applied is False
    assert report.gap_ha == pytest.approx(report.s0_vs_roks_ha - report.omega0_ha)
    saved = np.array(td.e)
    td.s0_t1_report(eq15=True, print_report=False)
    np.testing.assert_allclose(td.e, saved)


def test_unrestricted_reference_is_rejected(ch2_rohf):
    set_backend("cpu")
    uhf = scf.UHF(ch2_rohf.mol).run(conv_tol=1e-11)
    assert uhf.converged
    td = XSF_TDA_down(uhf, method=2, SA=0, davidson=False)
    td.kernel(nstates=2)
    with pytest.raises(NotImplementedError):
        td.s0_t1_report(print_report=False)
