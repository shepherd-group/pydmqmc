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
    """Tests __init__()"""

    sys = MatrixHamiltonian(input_file)
    H = sys.hamiltonian

    # Check the Hamiltonian.
    assert H.shape == (20,20)
    assert np.allclose(np.diag(H), known_diag)

    # Check derived quantities about the Hamiltonian.
    assert sys.n_determinants == 20
    assert sys.ref_energy == known_diag[0]

def test_MatrixHamiltonian_zero_hamiltonian(input_file, known_diag):
    """Tests zero_hamiltonian inherited from System."""

    sys = MatrixHamiltonian(input_file)
    sys.zero_hamiltonian()
    H = sys.hamiltonian

    ref = known_diag - sys.ref_energy

    assert np.allclose(np.diag(H), ref)
