import numpy as np
from pytest import fixture
from os.path import dirname, join

from pydmqmc.methods import FullConfigurationInteraction
from pydmqmc.systems import MatrixHamiltonian, Integral


@fixture
def matrix_system(request) -> MatrixHamiltonian:
    file = join(dirname(request.path),
                "..", "inputs", "hamiltonians", "EQUILIBRIUM-H4-STO3G.hamil")
    sys = MatrixHamiltonian(file)
    return sys


@fixture
def integral_system(request) -> Integral:
    file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
    sys = Integral(file)
    return sys


def test_FCI_MatrixHamiltonian(matrix_system):
    ref_eng = np.array(
        [-0.06111233,  0.43764421,  0.51264064,  0.82875661,  0.90125552,
        1.02634092,  1.04463431,  1.25803276,  1.45811183,  1.52207323,
        1.59932352,  1.6881537 ,  1.77427145,  2.02779824,  2.11131293,
        2.1947731 ,  2.47468662,  2.55639377,  2.84021179,  3.11960965]
    )

    mthd = FullConfigurationInteraction(matrix_system)
    mthd.start()

    assert np.allclose(mthd.energies, ref_eng)
    assert mthd.wavefunctions.shape == (20, 20)


def test_FCI_Integral(integral_system):
    ref_eng = np.array([-1.13728383,  0.48314267])
    ref_wav = np.array([[-0.99364675, -0.11254389],
                        [ 0.11254389, -0.99364675]])

    mthd = FullConfigurationInteraction(integral_system)
    mthd.start()

    assert np.allclose(mthd.energies, ref_eng)
    assert np.allclose(mthd.wavefunctions, ref_wav)
