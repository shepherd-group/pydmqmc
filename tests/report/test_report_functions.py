from pytest import fixture
import numpy as np

from pydmqmc.report.report_functions import *

class TestReportFunctions():

    @fixture(autouse=True)
    def _setup(self):
        self._matrix = np.arange(1,26).reshape((5,5))
        self._hamiltonian = np.flip(self._matrix)

    def test_trace(self):
        assert np.isclose(trace(self._matrix), 65)

    def test_energy_numerator(self):
        assert np.isclose(energy_numerator(self._matrix, self._hamiltonian),
                          3725)
    

    def test_energy_expectation(self):
        assert np.isclose(energy_expectation(self._matrix, self._hamiltonian),
                          3725/65)

    def test_von_neumann_numerator(self):
        assert np.isclose(von_neumann_numerator(self._matrix), -814.6906867)

    def test_von_neumann_expectation(self):
        assert np.isclose(von_neumann_expectation(self._matrix), -12.5337028)

