import numpy as np
from pytest import fixture
from os.path import dirname, join

from pydmqmc.systems import read_matrix, MatrixHamiltonian

@fixture
def file_equil_h4_sto3g(request):
    file = join(dirname(request.path),
                "..", "inputs", "hamiltonian", "EQUILIBRIUM-H4-STO3G.hamil")
    return file

@fixture
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

@fixture
def sorted_diag(known_diag):
    return np.sort(known_diag)

def test_read_matrix(file_equil_h4_sto3g, known_diag):

    ham = read_matrix(file_equil_h4_sto3g, is_complex=False)

    # Check the Hamiltonian.
    assert ham.shape == (20,20)
    assert np.allclose(np.diag(ham), known_diag)

def test_MatrixHamiltonian_init(file_equil_h4_sto3g, known_diag):

    sys = MatrixHamiltonian(file_equil_h4_sto3g)
    ham = sys.raw_hamiltonian

    # Check the Hamiltonian.
    assert ham.shape == (20,20)
    assert np.allclose(np.diag(ham), known_diag)

    # Check derived quantities about the Hamiltonian.
    assert sys.ndeterminants == 20
    assert sys.ref_energy == known_diag[0]

def test_MatrixHamiltonian_initialize(file_equil_h4_sto3g, sorted_diag):

    shift = 2
    sys = MatrixHamiltonian(file_equil_h4_sto3g, shift=shift, use_ip=True)

    # Test that Hamiltonian has been sorted correctly.
    # A successful sort implicitly verifies that the sort_map
    # attribute is accurate.
    diag = np.diag(sys.unshifted_hamiltonian)
    assert np.allclose(diag, sorted_diag)

    # Test that Hamiltonian has been shifted correctly 
    # by undoing the calculation.
    H = sys.hamiltonian
    II = np.eye(sys.ndeterminants)
    target = H + sys.ref_energy*II + shift*II
    assert np.allclose(target, sys.unshifted_hamiltonian)

    # Test that the non-interacting Hamiltonian was constructed correctly
    nH = sys.noninteracting_hamiltonian
    assert nH.shape == H.shape
    assert np.allclose(np.diag(nH), np.diag(H))
    # To check the off diagonal values are zero, mask out nonzero values
    # and compare the mask to the identity matrix.
    # Note that due to the shifting that subtracts off the reference energy,
    # index [0, 0] will be zero. Manually mask that value for easier checking.
    diag_mask = np.ma.masked_array(nH, mask = nH!=0)
    diag_mask.mask[0, 0] = True
    assert np.allclose(diag_mask.mask, II)
