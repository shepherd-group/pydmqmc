#!/usr/bin/env python

from .report import Report

from numpy.typing import NDArray as Array


class Eigenvalues(Report):
    r''' TODO: Write class docstring here.
    '''
    def __init__(
            self,
            **kwargs,
        ) -> None:
        r''' TODO: Write __init__ docstring here.
        '''
        Report.__init__(self, **kwargs)
        return

    def finish(self) -> None:
        r''' TODO: Write report docstring here.
        '''
        ndets = self.energies.shape[0]

        print(
            f'{"State":>12} {"Ei":>22} '
            + ''.join([f'{"C" + str(j+1):>12}' for j in range(ndets)])
        )

        for i in range(ndets):
            state = i + 1
            Ei = self.energies[i]
            Psii = self.wavefunctions[:, i]
            Psistr = self.wavefunction_string(Psii)

            print(
                f'{state:>12d} '
                f'{Ei:> 22.12f} '
                f'{Psistr:>120}'
            )

        return

    @staticmethod
    def wavefunction_string(
            wavefunction: Array,
            fstring: str = '> 12.6f',
        ) -> str:
        r''' TODO: Write wavefunction_string docstring here.
        '''
        s = ''.join([f'{Cj:{fstring}}' for Cj in wavefunction])

        return s
