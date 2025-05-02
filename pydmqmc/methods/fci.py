#!/usr/bin/env python

from .method import Analytic
from .. import systems

from numpy.linalg import eigh


class FullConfigurationInteraction(Analytic):
    """
    Perform full configuration interaction (FCI) for a given Hamiltonian.

    FCI is accomplished by performing exact diagonalization.

    Warnings
    --------
    The supplied system must have a hermtian Hamiltonian.
    """

    def __init__(
            self,
            system: systems.System
        ) -> None:
        super().__init__(system)

        # Prepare the system, if needed
        if self.system.hamiltonian is None:
            print("Generating Hamiltonian.")
            self.system.generate_hamiltonian()

        self._energies = None
        self._wavefunctions = None

        return

    @property
    def energies(self):
        """FCI energies."""
        return self._energies

    @property
    def wavefunctions(self):
        """FCI wavefunctions."""
        return self._wavefunctions

    def run(self) -> None:
        """TODO: Write start docstring here."""
        self._energies, self._wavefunctions = eigh(self.system.hamiltonian)
        return
