"""Base classes for the `methods` submodule."""

from .. import systems
from .. import utils
from ..report.registry import report_registry

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
        self._ran_calculation = False

    @property
    def system(self) -> systems.System:
        """The System object to which this method is applied."""
        return self._system

    @property
    def ran_calculation(self) -> bool:
        """Whether or not the calculation has been run."""
        return self._ran_calculation

    def run(self) -> None:
        """
        Base method for initializing iterative calculations.

        This base method handles a flag to ensure ``run`` is
        only called once. Any other calculations
        must be defined by the child class.
        """
        if self._ran_calculation:
            raise RuntimeError("A calculation has already been run for this "
                               "Method object. To prevent data loss, please "
                               "create a new Method object.")
        self._ran_calculation = True


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
        self._report_values = None
        self._report_reqs = None
        self._report_data = None

    @property
    def report_values(self) -> list[str] | None:
        """The list of values to be reported throughout the calculation."""
        return self._report_values

    @property
    def report_requirements(self) -> dict | None:
        """Dictionary of fulfilled requirements for each report value."""
        return self._report_reqs

    @property
    def report(self) -> list[dict] | None:
        """List of dictionaries with report values."""
        return self._report_data

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
        if self._ran_calculation:
            raise RuntimeError("A calculation has already been run for this "
                               "Method object. To prevent data loss, please "
                               "create a new Method object.")

        self._report_values = []
        self._report_reqs = {}

        for item in report_values:

            if item not in report_registry:
                raise AttributeError(f"Value {item} is not present in "
                                     "pydmqmc.report_registry. Did you "
                                     "forget to enroll it?")

            self._report_values.append(item)
            self._report_reqs[item] = report_registry.get_requirements(item,
                                                                       self)

        self._report_data = []

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
