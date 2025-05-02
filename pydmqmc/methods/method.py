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
        # Consider putting generate Hamiltonian portion here.
        # Unless not every method needs a Hamiltonian...?
        # But if every system needs a Hamiltonian,
        # Integral should just always generate it
        # (if it's too computationally intensive, an alternate
        # method can be added later)
        return

    def run(self) -> None:
        """TODO: Write run docstring here."""
        raise NotImplementedError(
            'The run function is not currently implemented; '
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

    def setup(self) -> None:
        """TODO: Write setup docstring here."""
        raise NotImplementedError(
            'The setup function is not currently implemented; '
            'please check your method or send patches!'
        )

        return