from .. import systems

class Method:
    """
    Base class for defining calculation methods.

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    """

    def __init__(
            self,
            system: systems.System,
            ) -> None:
        self.system = system
        return

    def start(self) -> None:
        """TODO: Write start docstring here."""
        raise NotImplementedError(
            'The start function is not currently implemented; '
            'please check your method or send patches!'
        )

        return


class Analytic(Method):
    """
    Base class for analytic calculations.
    
    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    """

    def __init__(self, 
                 system: systems.System):
        super().__init__(system)

        # consider putting generate hamiltonian portion here


class Iterative(Method):
    """
    Base class for iterative calculations.
    
    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    """

    def __init__(self, system: systems.System):
        super().__init__(system)

    def iterate(self) -> None:
        """TODO: Write iterate docstring here."""
        raise NotImplementedError(
            'The iterate function is not currently implemented; '
            'please check your method or send patches!'
        )

        return