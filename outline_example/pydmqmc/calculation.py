#!/usr/bin/env python

from .method import Method
from .system import System
from .report import Report


class Calculation(
        Method,
        System,
        Report,
    ):
    r''' TODO: Write class docstring here.
    '''
    def __init__(
            self,
            **kwargs,
        ) -> None:
        r''' TODO: Write __init__ docstring here.
        '''
        Method.__init__(self, **kwargs)
        System.__init__(self, **kwargs)
        Report.__init__(self, **kwargs)

        return

    def run(self) -> None:
        r''' TODO: Write run docstring here.
        '''
        self.start()

        while self.iterating:
            self.iterate()

        self.finish()

        return

    def iterate(self) -> None:
        r''' TODO: Write iterate docstring here.
        '''
        raise NotImplementedError(
            'The iterate function is not currently implemented '
            'please check your method_class or send patches!'
        )

        return
