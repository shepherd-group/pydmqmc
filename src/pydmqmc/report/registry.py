"""Class for associating strings with functions that calculate observables."""

from .report_functions import (
    trace,
    energy_numerator,
    energy_expectation,
    von_neumann_numerator,
    von_neumann_expectation,
)
from collections.abc import Iterable, Callable
from functools import partial


class _ReportRegistry:
    def __init__(self):
        self._registry = {
            "trace": trace,
            "energy numerator": energy_numerator,
            "energy expectation": energy_expectation,
            "von Neumann numerator": von_neumann_numerator,
            "von Neumann expectation": von_neumann_expectation,
        }
        # abbreviated list of requirements
        self._requirements = {
            "energy numerator": ("hamiltonian",),
            "energy expectation": ("hamiltonian",),
        }

    @property
    def functions(self) -> dict[str, Callable]:
        return self._registry

    @property
    def requirements(self) -> dict[str, tuple[str]]:
        return self._requirements

    def __len__(self):
        return len(self._registry)

    def __contains__(self, name):
        return name in self._registry

    def __getitem__(self, name):
        """Access the associated function."""
        return self._registry[name]

    def keys(self):
        return self._registry.keys()

    def list_requirements(self, name):
        try:
            req = self._requirements[name]
        except KeyError:
            req = None
        return req

    def enroll(
        self, name: str, function: Callable, requires: str | Iterable[str] | None
    ) -> None:
        """
        Include a new analysis function in the registry.

        Parameters
        ----------
        name : str
            Name by which to identify the function.
        function : function
            Function for performing the analysis. Must take a Method
            object as its first argument.
        requires : string, list of strings or None
            List of prerequisites required by the analysis function.
            These should be attributes of a Method or System object.
            Can be formatted as, e.g., `"method.system.hamiltonian"` or just
            `"hamiltonian"`. In the latter case, the Method and its System
            will be searched for an attribute called `hamiltonian`.

        Examples
        --------
        Assuming the existance of a user defined function called `my_func` that
        operates on a matrix in conjunction with a System's hamiltonian,
        we can enroll it into the global registry
        under the name `"my_analysis"` as below:
        >>> def my_func(matrix: NDArray, hamiltonian: NDArray):
        ...     return np.norm(hamiltonian @ matrix)
        >>> my_reg.enroll("my_analysis", my_func, ["hamiltonian"])
        """
        if name in self._registry:
            raise RuntimeError(
                f"A function called '{name}' is already enrolled in this registry!"
            )
        self._registry[name] = function

        if requires:
            self._requirements[name] = tuple(requires)  # make immutable

    def get_requirements(
        self, name: str, method: "Method"
    ) -> bool:  # avoid circular imports
        """
        Check for a function's requirements in a given Method.

        Will search both the supplied Method and its associated
        System object for the requirements associated with the
        analysis function `name`.

        Parameters
        ----------
        name : str
            Name of the analysis function for which we're checking
            requirements.
        method : Method object
            Method (and associated System) which we're checking.

        Returns
        -------
        bool
            Whether or not the analysis function's requirements are
            met by the given Method object.
        """
        if name not in self._registry:
            raise RuntimeError(
                f"Function '{name}' has not been enrolled in this registry!"
            )

        if name not in self._requirements:
            return {}

        req_dict = {}
        for req in self._requirements[name]:
            error_msg = (
                f"Requirements for {req} not currently "
                "satisfied. This Method and its System "
                "must be able to satisfy the following:"
                f" {self.list_requirements(req)}"
            )

            # The requirement may be the full attribute "path"
            try:
                req_data = eval(str(req))
                if req_data is None:
                    # requirement present but currently None
                    raise RuntimeError(error_msg)
                req_dict[req] = req_data  # add to return dict

            except NameError:
                # Check under the Method object
                test_str = "method." + req
                try:
                    req_data = eval(test_str)
                    if req_data is None:
                        # requirement present but currently None
                        raise RuntimeError(error_msg)
                    req_dict[req] = req_data  # add to return dict

                except AttributeError:
                    # Check under the System class
                    test_str2 = "method.system." + req
                    try:
                        req_data = eval(test_str2)
                        if req_data is None:
                            # requirement present but currently None
                            raise RuntimeError(error_msg)
                        req_dict[req] = req_data  # add to return dict
                    except AttributeError:
                        raise ValueError(
                            f"The requirement {req} is not "
                            "present in either the provided "
                            "Method or its associated System."
                        )

        # None of the requirements resulted in errors;
        # all could be found and weren't None
        return req_dict


# Create the main registry of report functions
report_registry = _ReportRegistry()


class enroll:
    """Decorator accepting keyword arguments for report_registry.enroll()."""

    # Taken from the yt-project's derived_field decorator :)

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def __call__(self, f: Callable) -> Callable:
        """Enroll a function in the report registry."""
        if "name" not in self._kwargs:
            self._kwargs["name"] = f.__name__
        partial(report_registry.enroll, function=f)(**self._kwargs)
        return f
