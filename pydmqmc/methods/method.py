from .. import systems

class Method:
    """Base class for defining calculation methods."""

    def __init__(
            self,
            system: systems.System,
            **kwargs,
            ) -> None:
        
        return

    def start(self) -> None:
        """TODO: Write start docstring here."""
        raise NotImplementedError(
            'The start function is not currently implemented; '
            'please check your method or send patches!'
        )

        return


class Analytic(Method):
    """Base class for analytic calculations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Iterative(Method):
    """Base class for iterative calculations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def iterate(self) -> None:
        """TODO: Write iterate docstring here."""
        raise NotImplementedError(
            'The iterate function is not currently implemented; '
            'please check your method or send patches!'
        )

        return