"""Base classes for the `methods` submodule."""

from .. import systems
from .. import utils
from ..report.registry import report_registry
from ..utils import save_report, ParallelHelper

import numpy as np

from collections.abc import Callable
from numpy.typing import ArrayLike


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
        Ensure calculations are only run once.

        This base method handles a flag to ensure ``run`` is
        only called once. Any other calculations
        must be defined by the child class.
        """
        if self._ran_calculation:
            raise RuntimeError(
                "A calculation has already been run for this "
                "Method object. To prevent data loss, please "
                "create a new Method object."
            )
        self._ran_calculation = True


class Analytic(Method):
    """
    Base class for analytic calculations.

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    """

    def __init__(self, system: systems.System):
        super().__init__(system)


class Iterative(Method):
    """
    Base class for iterative calculations.

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    rng_seed : int or array_like of ints, optional
        Seed or sequence of seeds for the psuedo-random number generator.
        See :func:`numpy.random.default_rng`. If using MPI parallelization,
        each processor will have a unique seed based on this value.
    parallel : bool, default False
        Whether to use MPI to parallelize the calculation.
    """

    def __init__(
        self,
        system: systems.System,
        rng_seed: None | int | ArrayLike = None,
        parallel: bool = False,
    ):
        super().__init__(system)

        self._report_quants = None
        self._report_reqs = None
        self._report_data = None

        self._parallel: bool = parallel
        self._ph: ParallelHelper | None = None

        # The ParallelHelper must be initialized by the child class
        # since the child method will know how to distribute the problem.
        # Initialization of the rng seed (i.e. a call to reset_rng) must follow.

    @property
    def report_values(self) -> list[str] | None:
        """The list of values to be reported throughout the calculation."""
        return self._report_quants

    @property
    def report_requirements(self) -> dict | None:
        """Dictionary of fulfilled requirements for each report value."""
        return self._report_reqs

    @property
    def report(self) -> list[dict] | None:
        """List of dictionaries with report values."""
        return self._report_data

    @property
    def parallel(self) -> bool:
        """Whether this method is set up to run in parallel."""
        return self._parallel

    @property
    def parallel_size(self) -> int | None:
        """Number of processors if running in parallel."""
        if self._parallel and self._ph is not None:
            return self._ph.size
        else:
            return None

    @property
    def parallel_rank(self) -> int | None:
        """Rank of the current processor if running in parallel."""
        if self._parallel and self._ph is not None:
            return self._ph.rank
        else:
            return None

    @property
    def parallel_is_parent(self) -> bool | None:
        """Whether the current processor is the root if running in parallel."""
        if self._parallel and self._ph is not None:
            return self._ph.parent
        else:
            return None

    @property
    def is_reporter(self) -> bool:
        """
        Whether the current process is the designated reporter.

        This process creates the iteration report and writes it to both
        stdout and to file. If running in serial, the current process is
        always designated as the reporter.
        """
        if self._parallel:
            if self._ph is not None:
                return self._ph.parent
            else:
                # ParallelHelper has not been initialized yet
                # This case shouldn't really happen outside of the test suite
                return True
        else:
            return True

    def reset_rng(self, rng_seed: None | int | ArrayLike = None) -> None:
        """
        Create a new psuedo-random number generator with the given seed.

        If running in parallel, each processor will have a unique seed
        based on the supplied `rng_seed`.

        Parameters
        ----------
        rng_seed : int or array_like of ints, optional
            Seed or sequence of seeds for the psuedo-random number generator.
            See :func:`numpy.random.default_rng`
        """
        if self._parallel:
            seed = self._ph.get_rng_seed(rng_seed)
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(rng_seed)

    def setup(self, report_values: list[str]) -> None:
        """
        Create structures for periodically reporting calculation state.

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
            raise RuntimeError(
                "A calculation has already been run for this "
                "Method object. To prevent data loss, please "
                "create a new Method object."
            )

        self._report_quants = []
        self._report_reqs = {}

        for item in report_values:
            if item not in report_registry:
                raise AttributeError(
                    f"Value {item} is not present in "
                    "pydmqmc.report_registry. Did you "
                    "forget to enroll it?"
                )

            self._report_quants.append(item)
            self._report_reqs[item] = report_registry.get_requirements(item, self)

        self._report_data = []

    def save_data(  # TODO test this function, ideally here but maybe in Child tests
        self,
        report_basename: str,
        report_filetype: str = "csv",
        pickle_protocol: int | None = None,
        index_quant: str | None = None,
    ):
        """
        Save the iteration report to file.

        Parameters
        ----------
        report_basename : str
            Base filename for saving the iteration report. Will automatically
            be combined with the extension specified by ``report_filetype``.
        report_filetype : str, default "csv"
            File type (aka extension) with which to save the report.
            Supported types are:

            - "csv" : comma-separated value file
            - "txt" : text file (space-delimited)
            - "pkl" : pickle file

        pickle_protocol : unt, optional
            Protocol version to use if either `filetype` is "pkl".
            If none, uses `pickle`'s default.
        index_quant : string, optional
            Name of the quantity to put as the first column.
            Must be a key in the report.
        """
        if self.is_reporter:
            save_report(
                self._report_data,
                report_basename + "_report",
                index_quant,
                report_filetype,
                pickle_protocol,
            )

    def parse_method(self, method: str = "euler") -> Callable:
        """
        Parse the supplied string to return the corresponding function.

        Expected call signature for each method is
        ``(func, x, y, dy)`` where func(x, y) = dx/dy.

        Supported methods are:
            * Forward Euler (``euler``)
            * 4th Order Runge Kutta (``rk4``)

        Parameters
        ----------
        method : {"euler", "rk4"}
            Desired integration method.

        Returns
        -------
        callable
            The associated function from :mod:`pydmqmc.utils.integrators`
        """
        method = method.lower()
        if not self.parallel:
            if method == "euler":
                return utils.euler
            elif method == "rk4":
                return utils.rk4
            else:
                raise RuntimeError(f"Update method {method} is not recognized.")
        else:
            if method == "euler":
                return utils.parallel_euler
            elif method == "rk4":
                return utils.parallel_rk4
            else:
                raise RuntimeError(
                    f"Parallel update method {method} is not recognized."
                )
