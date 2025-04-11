#!/usr/bin/env python


class Calculation:
    ...


def setup(
        method: object = ...,
        system: object = ...,
        report: object = ...,
        **kwargs,
    ) -> Calculation:
    r''' TODO: Write setup docstring here.
    '''

    class UserCalculation(
            method,
            system,
            report,
        ):
        r''' TODO: Write class docstring here.
        '''
        def __init__(
                self,
                **kwargs,
            ) -> None:
            r''' TODO: Write __init__ docstring here.
            '''
            method.__init__(self, **kwargs)
            system.__init__(self, **kwargs)
            report.__init__(self, **kwargs)

            return

        def run(self) -> None:
            r''' TODO: Write run docstring here.
            '''
            self.start()

            while self.iterating:
                self.iterate()

            self.finish()

            return

    return UserCalculation(**kwargs)
