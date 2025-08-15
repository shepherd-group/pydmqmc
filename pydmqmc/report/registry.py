from .report_functions import *

from collections.abc import Iterable, Callable
from functools import partial

class ReportRegistry():
    def __init__(self):
        self._registry = {
            'trace': trace,
            'energy': energy,
            'von Neumann':von_neumann,
            }
        self._requirements = {
            'energy': ('hamiltonian',),
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
        """A shortcut for accessing the associated function."""
        return self._registry[name]

    def keys(self):
        return self._registry.keys()

    def enroll(self,
               name: str,
               function: Callable,
               requires: str | Iterable[str] | None) -> None:
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

        Examples
        --------
        Assuming the existance of a user defined function called `my_func` that
        operates on a matrix, we can enroll it into the global registry
        under the name `"my_analysis"` as below. 
        >>> def energy(matrix: NDArray, hamiltonian: NDArray):
        ...     return np.norm(hamiltonian @ matrix)
        >>> my_reg.enroll("my_analysis",
        ...               my_func,
        ...               ["hamiltonian"])
        """
        if name in self._registry:
            raise RuntimeError(f"A function called '{name}' is "
                               "already enrolled in this registry!")
        self._registry[name] = function

        if requires:
            self._requirements[name] = tuple(requires)  # make immutable
        else:
            # If requires is an empty list or string, also set to None
            self._requirements[name] = None

    def check_requirements(self,
                           name: str,
                           method: 'Method') -> bool:
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
            raise RuntimeError(f"Function '{name}' has not been "
                               "enrolled in this registry!")

        for req in self._requirements[name]:
            test_str = "method." + req
            try:
                exec(test_str)
            except AttributeError:
                # This requirement could also be part of the
                # system class, so check there too
                test_str2 = "method.system." + req
                try:
                    exec(test_str2)
                except AttributeError:
                    return False

        # None of the requirements resulted in errors
        return True

# Create the main registry of report functions
report_registry = ReportRegistry()

class enroll:
    # Implement a decorator accepting keyword arguments
    # that are passed report_registry.enroll()
    # Taken from the yt-project's derived_field decorator :)

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def __call__(self, f: Callable) -> Callable:
        if "name" not in self._kwargs:
            self._kwargs["name"] = f.__name__
        partial(report_registry.enroll, function=f)(**self._kwargs)
        return f
