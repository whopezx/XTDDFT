"""Opt-in omega0 for XSF-TDA-down on triplet CH2 / STO-3G."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pytest
from pyscf import dft, gto, lib, scf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.utils.backend import set_backend
from XTDDFT.utils.unit import ha2eV


# Triplet geometry with closed, open, and virtual orbital blocks.
ATOM = (
    ("C", (0.00, 0.00, 0.00)),
    ("H", (0.00, 0.95, 0.15)),
    ("H", (0.10, -0.75, 0.65)),
)


@lru_cache(maxsize=8)
def _reference(kind, spin, xc):
    set_backend("cpu")
    lib.num_threads(2)
    mol = gto.M(
        atom=ATOM,
        basis="sto-3g",
        spin=spin,
        charge=0,
        unit="Angstrom",
        verbose=0,
    )
    if kind == "rohf":
        mf = scf.ROHF(mol)
    elif kind == "rhf":
        mf = scf.RHF(mol)
    elif kind == "uhf":
        mf = scf.UHF(mol)
    elif kind == "roks":
        mf = dft.ROKS(mol)
        mf.xc = xc
        mf.grids.level = 1
    elif kind == "uks":
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.grids.level = 1
    else:
        raise ValueError(kind)
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return mf


def reference(kind, spin=2, xc="svwn"):
    return deepcopy(_reference(kind, spin, xc))


def assert_vectors_match(left, right, atol=1e-8):
    """Eigenvectors match up to the arbitrary sign from the eigensolver."""
    left = np.asarray(left)
    right = np.asarray(right)
    assert left.shape == right.shape
    aligned = np.array(right, copy=True)
    for column in range(right.shape[1]):
        if np.dot(left[:, column], right[:, column]) < 0.0:
            aligned[:, column] *= -1.0
    np.testing.assert_allclose(left, aligned, rtol=0, atol=atol)


def eq12_vector(td):
    """x(mα → nβ) = δ_mn / sqrt(n_open) in the full CV|CO|OV|OO space."""
    nc, no, nv = int(td.nc), int(td.no), int(td.nv)
    dim3 = nc * nv + nc * no + no * nv
    vector = np.zeros(dim3 + no * no)
    vector[dim3 + np.arange(no) * no + np.arange(no)] = 1.0 / np.sqrt(no)
    return vector


def test_eq12_vector_is_normalized_oo_diagonal():
    td = XSF_TDA_down(reference("rohf"), method=2, SA=3, davidson=True)
    assert td.nc >= 1 and td.no >= 1 and td.nv >= 1
    x = eq12_vector(td)
    nc, no, nv = int(td.nc), int(td.no), int(td.nv)
    dim3 = nc * nv + nc * no + no * nv
    np.testing.assert_allclose(x, td._intra_multiplet_vector(), rtol=0, atol=0)
    assert np.count_nonzero(x[:dim3]) == 0
    oo = x[dim3:].reshape(no, no)
    np.testing.assert_allclose(np.diag(oo), 1.0 / np.sqrt(no))
    np.testing.assert_allclose(oo - np.diag(np.diag(oo)), 0.0, atol=0)
    np.testing.assert_allclose(np.dot(x, x), 1.0)
    # This is the OO direction get_vect()/remove() project out.
    np.testing.assert_allclose(td.get_vect().T @ x[dim3:], 0.0, atol=1e-12)


@pytest.mark.parametrize("davidson", [True, False])
def test_rohf_omega0_is_zero_and_roots_unchanged(davidson, capsys):
    mf = reference("rohf")
    kwargs = dict(method=2, SA=3, davidson=davidson, davidson_backend="cpu")
    plain = XSF_TDA_down(mf, **kwargs)
    energies, vectors = plain.kernel(nstates=3)
    assert "omega0" not in capsys.readouterr().out
    assert not hasattr(plain, "omega0")
    saved_e = np.array(np.asarray(plain.e), copy=True)
    saved_v = np.array(np.asarray(plain.v), copy=True)
    # Evaluating omega0 must not rewrite the roots already stored by kernel().
    plain._evaluate_omega0()
    np.testing.assert_array_equal(np.asarray(plain.e), saved_e)
    np.testing.assert_array_equal(np.asarray(plain.v), saved_v)
    assert abs(plain.omega0) < 1e-8

    opted = XSF_TDA_down(mf, **kwargs)
    energies_w, vectors_w = opted.kernel(nstates=3, omega0=True)
    printed = capsys.readouterr().out
    assert "omega0" in printed
    assert abs(opted.omega0) < 1e-8
    np.testing.assert_allclose(opted.omega0_ev, opted.omega0 * ha2eV, rtol=0, atol=0)
    # Two diagonalizations agree to roundoff; omega0 is applied after self.e is stored.
    np.testing.assert_allclose(energies, energies_w, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(plain.e), np.asarray(opted.e), rtol=0, atol=1e-12)
    assert_vectors_match(vectors, vectors_w)
    full = (opted.nc + opted.no) * (opted.no + opted.nv)
    assert np.asarray(vectors_w).shape[0] == full - 1
    if davidson:
        assert np.all(opted.converged)


@pytest.mark.parametrize(
    "kind,method",
    [
        ("rohf", 2),
        ("roks", 0),  # ALDA0
        ("roks", 1),  # multicollinear
        ("roks", 2),  # collinear
    ],
)
def test_a_minus_matvec_matches_dense_sa0(kind, method):
    mf = reference(kind)
    samples = 20
    solver = XSF_TDA_down(
        mf, method=method, SA=3, davidson=True,
        davidson_backend="cpu", collinear_samples=samples,
    )
    assert solver.SA == 3 and solver.re
    vind, hdiag = solver.gen_tda_operation_sf(use_delta_a=False, project_oo=False)
    assert solver.SA == 3 and solver.re

    dense = XSF_TDA_down(
        reference(kind), method=method, SA=0, davidson=False,
        collinear_samples=samples,
    )
    matrix = np.asarray(dense._prepare_dense_A(fglobal=1.0))
    full = matrix.shape[0]
    assert not dense.re
    assert matrix.shape == (full, full)
    assert np.asarray(hdiag).shape == (full,)

    x = eq12_vector(solver)
    probes = np.vstack([
        x,
        np.random.default_rng(19).normal(size=(2, full)),
    ])
    got = np.asarray(vind(probes))
    np.testing.assert_allclose(probes @ matrix.T, got, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        probes @ np.asarray(dense.sf_tda_A).T, got, rtol=0, atol=1e-8,
    )


@pytest.mark.parametrize("davidson", [False, True])
def test_alda0_kernel_omega0_matches_unprojected_a_minus(davidson):
    mf = reference("roks")
    dense = XSF_TDA_down(mf, method=0, SA=0, davidson=False)
    matrix = np.asarray(dense._prepare_dense_A(fglobal=1.0))
    solved = reference("roks")
    td = XSF_TDA_down(
        solved, method=0, SA=3, davidson=davidson, davidson_backend="cpu",
    )
    bare_e, bare_v = td.kernel(nstates=2)
    assert not hasattr(td, "omega0")

    opted = XSF_TDA_down(
        solved, method=0, SA=3, davidson=davidson, davidson_backend="cpu",
    )
    energies, vectors = opted.kernel(nstates=2, omega0=True)
    x = eq12_vector(opted)
    np.testing.assert_allclose(opted.omega0, float(x @ matrix @ x), rtol=0, atol=1e-8)
    np.testing.assert_allclose(energies, bare_e, rtol=0, atol=1e-12)
    assert_vectors_match(vectors, bare_v)
    assert np.asarray(vectors).shape[0] == matrix.shape[0] - 1


def test_omega0_rejects_unrestricted_and_closed_shell():
    uks = XSF_TDA_down(reference("uks"), method=0, SA=0, davidson=True)
    with pytest.raises(NotImplementedError, match="UKS/UHF"):
        uks.kernel(nstates=1, omega0=True)

    uhf = XSF_TDA_down(reference("uhf"), method=2, davidson=False)
    with pytest.raises(NotImplementedError, match="UKS/UHF"):
        uhf.kernel(nstates=1, omega0=True)

    rhf = XSF_TDA_down(reference("rhf", spin=0), method=2, davidson=False)
    with pytest.raises(ValueError, match="open-shell"):
        rhf.kernel(nstates=1, omega0=True)
