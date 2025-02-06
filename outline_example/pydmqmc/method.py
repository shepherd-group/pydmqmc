#!/usr/bin/env python

from .ci import FullConfigurationInteraction


class Method(
        FullConfigurationInteraction,
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
        FullConfigurationInteraction.__init__(self, **kwargs)

        return

    #def start(self) -> None:
    #    r''' TODO: Write start docstring here.
    #    '''
    #    raise NotImplementedError(
    #        'The start function is not currently implemented '
    #        'please check your method_class or send patches!'
    #    )

    #    return
