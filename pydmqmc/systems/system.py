
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
    def hamiltonian(self) -> None:
        """Placeholder for the system's Hamiltonian."""
        return None

    @property
    def n_determinants(self) -> None:
        """Placeholder for the size of the system's Hilbert space."""
