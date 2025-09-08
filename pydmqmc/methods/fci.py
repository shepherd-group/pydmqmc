"""Analytic FCI method."""

from .method import Analytic
from .. import systems
from ..utils import save_array

from numpy.linalg import eigh


class FullConfigurationInteraction(Analytic):
    """
    Perform full configuration interaction (FCI) for a given Hamiltonian.

    FCI is accomplished by performing exact diagonalization.

    Warnings
    --------
    The supplied system must have a hermtian Hamiltonian.
    """

    def __init__(
            self,
            system: systems.System
            ) -> None:
        super().__init__(system)

        # Prepare the system, if needed
        if self.system.hamiltonian is None:
            print("Generating Hamiltonian.")
            self.system.generate_hamiltonian()

        self._energies = None  # 1D array
        self._wavefunctions = None  # 2D array

        return

    @property
    def energies(self):
        """FCI energies."""
        return self._energies

    @property
    def wavefunctions(self):
        """FCI wavefunctions."""
        return self._wavefunctions

    def run(self) -> None:
        """TODO: Write start docstring here."""
        self._energies, self._wavefunctions = eigh(self.system.hamiltonian)
        return

    def save_data(self,
                  basename: str,
                  energy_filetype: str = "csv",
                  wavefunction_filetype: str = "csv",
                  pickle_protocol: int | None = None) -> None:
        """
        Save the final energies and wavefunctions to file.

        The `basename` and `filetype` parameters will be used to construct
        filenames for all of the data written to file. For example, if
        `basename` is "test_run" and the `energy_` and `wavefunction_filetype`
        are both "csv", the energies will be saved to
        "test_run_energies.csv" and the wavefunctions will be saved to
        "test_run_wavefunctions.csv".

        Parameters
        ----------
        basename : str
            Base name used to construct the filenames for the energies
            and wavefunctions
        energy_filetype, wavefunction_filetype : str, default "csv"
            File type (aka extension) with which to save the energies.
            Supported types are:

            - "csv" : comma-separated value file
            - "npy" : NumPy binary file
            - "pkl" : Python pickle file
            - "txt" : text file (space-delimited)

        pickle_protocol : unt, optional
            Protocol version to use if either `filetype` is "pkl".
            If none, uses `pickle`'s default.
        """
        save_array(self._energies,
            basename + "_energies",
            energy_filetype,
            pickle_protocol)
        
        save_array(self._wavefunctions,
            basename + "_wavefunctions",
            wavefunction_filetype,
            pickle_protocol)
