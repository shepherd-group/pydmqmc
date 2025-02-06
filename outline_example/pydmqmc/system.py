#!/usr/bin/env python

from .hamiltonian import MatrixHamiltonian


class System(
        MatrixHamiltonian,
    ):
    r''' TODO: Write class docstring here.
    '''
    def __init__(
            self,
            **kwargs,
        ) -> None:
        r''' TODO: Write __init__ docstring here.
        '''
        # TODO: Make this inheritence general.
        MatrixHamiltonian.__init__(self, **kwargs)

        return
