from pydmqmc.utils.symmetry import *


def test_orb_sym():
    # I'm replicating the random choice in Integral.random_bitarray_symspace
    sym = orb_sym([4, 0], 7)
    assert sym == 4

def test_pg_sym_cross_prod():
    prod = pg_sym_cross_prod(2, 4, 7)
    assert prod == 6

def test_pg_sym_conj():
    conj = pg_sym_conj(4, 3)
    assert conj == 0