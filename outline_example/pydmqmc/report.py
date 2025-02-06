#!/usr/bin/env python

from .eigenvalues import Eigenvalues


class Report(
        Eigenvalues,
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
        Eigenvalues.__init__(self, **kwargs)

        return
