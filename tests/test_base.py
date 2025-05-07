from pytest import fixture, raises

import pydmqmc.utils as utils
from pydmqmc.systems import System
from pydmqmc.methods import Method, Analytic, Iterative

@fixture
def dummy_system() -> System:
    sys = System("/dummy/path")
    return sys


@fixture
def dummy_iterative(dummy_system) -> Iterative:
    mtd = Iterative(dummy_system)
    return mtd


def test_System(dummy_system):
    assert dummy_system.input_file == "/dummy/path"
    assert dummy_system.is_complex == False
    assert dummy_system.ref_energy == 0.0
    assert dummy_system.hamiltonian is None
    assert dummy_system.n_determinants is None


def test_System_zero_hamiltonian(dummy_system):
    with raises(RuntimeError):
        dummy_system.zero_hamiltonian()


def test_Method(dummy_system):
    mtd = Method(dummy_system)

    assert id(mtd.system) == id(dummy_system)


def test_Method_run(dummy_system):
    mtd = Method(dummy_system)

    with raises(NotImplementedError):
        mtd.run()


def test_Analytic(dummy_system):
    mtd = Analytic(dummy_system)

    assert id(mtd.system) == id(dummy_system)


def test_Analytic_run(dummy_system):
    mtd = Analytic(dummy_system)

    with raises(NotImplementedError):
        mtd.run()


def test_Iterative(dummy_iterative, dummy_system):
    assert id(dummy_iterative.system) == id(dummy_system)


def test_Iterative_run(dummy_iterative):
    with raises(NotImplementedError):
        dummy_iterative.run()


def test_Iterative_setup(dummy_iterative):
    with raises(NotImplementedError):
        dummy_iterative.setup()


def test_Iterative_parse_method(dummy_iterative):
    f_euler = dummy_iterative.parse_method("euler")
    assert f_euler is utils.euler
