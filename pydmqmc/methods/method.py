"""Base classes for the `methods` submodule."""

from .. import systems
from .. import utils
from .. import report_registry

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
        self._report_data = None

    def setup(self, report_values) -> None:
        """
        Base method for initializing iterative calculations.

        This base method creates a data structure for reporting 
        user-supplied values every iteration. Any other setup
        activities must be defined by the child class.

        Parameters
        ----------
        report_values : list of strings
            List of values to periodically report while performing
            the calculation. Each item must be recognized by the
            `report_registry`. The iteration variable
            will automatically be included.
        """
        self._report_data = {}

        for item in report_values:

            if item not in report_registry:
                raise RuntimeError(f"Value {item} is not present in "
                                   "pydmqmc.report_registry. Did you "
                                   "forget to enroll it?")

            self._report_data[item] = []

    def parse_method(self, method: str = "euler") -> Callable:
        """
        Parse the supplied string to return the corresponding function.

        Call signature is func, x, y, dy where func(x, y) = dx/dy.
        TODO: list supported methods.
        """
        method = method.lower()
        if method == "euler":
            return utils.euler
        elif method == "rk4":
            return utils.rk4
        else:
            raise RuntimeError(f"Update method {method} is not recognized.")
