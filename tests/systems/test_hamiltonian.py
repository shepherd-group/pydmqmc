import pytest
import numpy as np
from os.path import dirname, join

from pydmqmc.systems import MatrixHamiltonian

@pytest.fixture
def known_diag():
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

def test_read_matrix(request, known_diag):

    file = join(dirname(request.path),
                "..", "inputs", "hamiltonian", "EQUILIBRIUM-H4-STO3G.hamil")
    sys = MatrixHamiltonian(file)
    ham = sys.hamiltonian

    # Check Hamiltonian
    assert ham.shape == (20,20)
    assert np.allclose(np.diag(ham), known_diag)

    # Check derived quantities
    assert sys.ndeterminants == 20
    assert sys.ref_energy == known_diag[0]
