import numpy as np
from pytest import fixture, raises
from os.path import dirname, join

from numpy.typing import NDArray as Array

from pydmqmc.systems import MatrixHamiltonian


@fixture
def input_file(request) -> str:
    file = join(dirname(request.path),
                "..", "inputs", "hamiltonians", "EQUILIBRIUM-H4-STO3G.hamil")
    return file


@fixture
def known_diag() -> Array:
    values = [
       -2.11534977839243    , -1.13379264192082    ,
       -1.48665751992125    , -0.644571454998723   ,
       -1.29173833412146    , -0.537126997125704   ,
       -1.13379264192082    , -0.04063719397208587 ,
       -0.559346283023184   ,  0.39433809337647    ,
       -1.48665751992125    , -0.559346283023184   ,
       -0.78502052330756    ,  0.002819642041427484,
       -0.537126997125704   ,  0.267078948771445   ,
       -0.644571454998723   ,  0.394338093376471   ,
        0.002819642041427484,  0.902258118867541   ]

    return np.array(values)


def test_MatrixHamiltonian_load_complex(input_file):

    with raises(NotImplementedError):
        MatrixHamiltonian(input_file, is_complex=True)


def test_MatrixHamiltonian_load(input_file, known_diag):
    """Tests __init__() before the call to _shift()."""

    sys = MatrixHamiltonian(input_file)
    raw = sys.unshifted_hamiltonian

    # Check the Hamiltonian.
    assert raw.shape == (20,20)
    assert np.allclose(np.diag(raw), known_diag)

    # Check derived quantities about the Hamiltonian.
    assert sys.ndeterminants == 20
    assert sys.ref_energy == known_diag[0]


def test_MatrixHamiltonian_shift_shift(input_file):
    """Tests _shift()."""

    shift = 2
    sys = MatrixHamiltonian(input_file, shift=shift)

    # Test that Hamiltonian has been shifted correctly 
    # by undoing the calculation.
    H = sys.hamiltonian
    II = np.eye(sys.ndeterminants)
    target = H + sys.ref_energy*II + shift*II
    assert np.allclose(target, sys.unshifted_hamiltonian)


def test_MatrixHamiltonian_shift_use_ip(input_file):
    """Tests _shift()."""

    sys = MatrixHamiltonian(input_file, use_ip=True)

    # Test that the non-interacting Hamiltonian was constructed correctly
    H = sys.hamiltonian
    nH = sys.noninteracting_hamiltonian
    assert nH.shape == H.shape
    assert np.allclose(np.diag(nH), np.diag(H))
    # To check the off diagonal values are zero, mask out nonzero values
    # and compare the mask to the identity matrix.
    # Note that due to the shifting that subtracts off the reference energy,
    # index [0, 0] will be zero. Manually mask that value for easier checking.
    diag_mask = np.ma.masked_array(nH, mask = nH!=0)
    diag_mask.mask[0, 0] = True
    assert np.allclose(diag_mask.mask, np.eye(sys.ndeterminants))
