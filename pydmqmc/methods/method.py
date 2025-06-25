"""Base classes for the `methods` submodule."""

from .. import systems
from .. import utils

from collections.abc import Callable


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
        self._system = system
        return

    @property
    def system(self) -> systems.System:
        """The System object to which this method is applied."""
        return self._system

    def run(self) -> None:
        """TODO: Write run docstring here."""
        raise NotImplementedError(
            f'The run method for {self.__class__.__name__} is not '
            'currently implemented; please check your method or send patches!'
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
            f'The setup method for {self.__class__.__name__} is not '
            'currently implemented; please check your method or send patches!'
        )

        return

    def parse_method(self, method: str = "euler") -> Callable:
        """
        Parse the supplied string to return the corresponding function.

        Call signature is func, x, y, dy where func(x, y) = dx/dy.
        I should list supported methods.
        """
        if method.lower() == "euler":
            return utils.euler
        else:
            raise RuntimeError(f"Update method {method} is not recognized.")
