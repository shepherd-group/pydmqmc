#!/usr/bin/env python

from .method import Analytic

from numpy.linalg import eigh


class FullConfigurationInteraction(Analytic):
    """TODO: Write class docstring here."""

    def __init__(
            self,
            **kwargs,
        ) -> None:
        super().__init__(self, **kwargs)
        return

    def start(self) -> None:
        """TODO: Write start docstring here."""
        self.energies, self.wavefunctions = eigh(self.hamiltonian)
        return
