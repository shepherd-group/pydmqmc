#!/usr/bin/env python

from .method import Method

from numpy.linalg import eigh


class FullConfigurationInteraction(Method):
    r''' TODO: Write class docstring here.
    '''
    iterating: bool = False

    def __init__(
            self,
            **kwargs,
        ) -> None:
        r''' TODO: Write __init__ docstring here.
        '''
        Method.__init__(self, **kwargs)
        return

    def start(self) -> None:
        r''' TODO: Write start docstring here.
        '''
        self.energies, self.wavefunctions = eigh(self.hamiltonian)
        return
