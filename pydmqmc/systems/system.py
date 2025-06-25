"""Base class for the `systems` submodule."""

from numpy import eye

from numpy.typing import NDArray as Array


class System:
    """
    Base class for defining quantum systems.

    Parameters
    ----------
    input_file : str
        Name of the integral file that defines the system.
    is_complex : bool, default True
        Whether or not the integral is complex;
        controls the integral index symmetry.
    """

    def __init__(
            self,
            input_file: str,
            is_complex: bool = False,
            **kwargs,
            ) -> None:

        self._input_file = input_file
        self._is_complex = is_complex

        self._ref_eng = 0.0
        self._H = None
        self._ndets = None

        self._nex_mat = None

        return

    @property
    def input_file(self) -> str:
        """Filename for loaded Hamiltonian."""
        return self._input_file

    @property
    def is_complex(self) -> bool:
        """Whether or not the Hamiltonain is complex."""
        return self._is_complex

    @property
    def ref_energy(self) -> float:
        """Reference Hartree-Fock energy."""
        return self._ref_eng

    @property
    def hamiltonian(self) -> None | Array:
        """The system's Hamiltonian."""
        return self._H

    @property
    def n_determinants(self) -> None | Array:
        """Size of the Hilbert space."""
        return self._ndets

    @property
    def excitation_matrix(self) -> Array:
        """An `n_determinants`-square matrix of excitations between i and j."""
        return self._nex_mat

    def zero_hamiltonian(self) -> None:
        """
        Subtract the Hartree-Fock energy from the Hamiltonian.

        This will overwrite the existing Hamiltonian with the
        shifted version.
        """
        if self._H is not None:
            self._H -= self._ref_eng * eye(self._ndets)
        else:
            raise RuntimeError(
                "The Hamiltonian is currently `None` and cannot be shifted.")

    def generate_excitation_matrix(self):
        """Replace with child-specific method."""
        raise NotImplementedError(f"{self.__class__.__name__} does not "
                                  "currently have a method for generating the "
                                  "excitation matrix. Please send patches!")
