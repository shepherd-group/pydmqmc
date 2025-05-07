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
        [-2.17646211, -1.67770557, -1.60270914, -1.28659316, -1.21409426,
       -1.08900886, -1.07071547, -0.85731702, -0.65723795, -0.59327654,
       -0.51602625, -0.42719608, -0.34107833, -0.08755154, -0.00403685,
        0.07942332,  0.35933684,  0.44104399,  0.72486201,  1.00425988]
    )

    mthd = FullConfigurationInteraction(matrix_system)
    mthd.run()

    assert np.allclose(mthd.energies, ref_eng)
    assert mthd.wavefunctions.shape == (20, 20)


def test_FCI_Integral(integral_system):
    ref_eng = np.array([-1.13728383,  0.48314267])
    ref_wav = np.array([[-0.99364675, -0.11254389],
                        [ 0.11254389, -0.99364675]])

    mthd = FullConfigurationInteraction(integral_system)
    mthd.run()

    assert np.allclose(mthd.energies, ref_eng)
    assert np.allclose(mthd.wavefunctions, ref_wav)
