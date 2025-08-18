"""Iterative DMQMC methods using symmetric & asymmetric Bloch equations."""

from .method import Iterative
from ..systems import System
from ..report.registry import report_registry
from ..utils import save_array, save_report

import numpy as np
from numba import njit

from numpy.typing import ArrayLike, NDArray as Array

import warnings  # remove when no longer in use


class DensityMatrixQMC(Iterative):
    """
    Density matrix quantum Monte Carlo.

    CK Note: rather than the proposed common start() and iterate() methods,
    I refactored the common methods to setup() and run().

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    rng_seed : int or array_like of ints, optional
        Seed or sequence of seeds for the psuedo-random number generator.
        See :func:`numpy.random.default_rng`
    """

    def __init__(
            self,
            system: System,
            rng_seed: None | int | ArrayLike = None
            ) -> None:
        super().__init__(system)

        # Prepare the system, if needed.
        if self.system.hamiltonian is None:
            print("Generating Hamiltonian.")
            self.system.generate_hamiltonian()

        self._rng = np.random.default_rng(rng_seed)

        self._density_matrix = None
        self._S = None

        return

    @property
    def density_matrix(self) -> None | Array:
        """Density matrix."""
        return self._density_matrix

    def reset_rng(self,
                  rng_seed: None | int | ArrayLike = None
                  ) -> None:
        """
        Create a new psuedo-random number generator with the given seed.

        Parameters
        ----------
        rng_seed : int or array_like of ints, optional
            Seed or sequence of seeds for the psuedo-random number generator.
            See :func:`numpy.random.default_rng`
        """
        self._rng = np.random.default_rng(rng_seed)

    def setup(self,
              initialization: str = "deterministic",
              n_particles: int = 1,
              fixed_diagonal: ArrayLike | None = None,
              report_values: list[str] = ["trace", "energy"]
              ) -> None:
        """
        Specify conditions for the DMQMC realization.

        This setup includes the initial density matrix and a data structure
        for reporting user-supplied values every iteration.

        Parameters
        ----------
        initialization : str, default "deterministic"
            Initialization method for the density matrix. See Notes for more.
            Must be one of:

            - deterministic
            - random-uniform
            - fixed

        n_particles : int, default 1
            The initial number of psip particles that should be present
            in the density matrix. Only used with the "deterministic" method.
        diag : array_like, optional
            Directly defined the diagonal of the density matrix when used
            with the "fixed" initialization method. The length of `diag`
            must be the same as the number of determinants in the system.
        report_values : list, optional
            List of values to periodically report while performing
            the calculation. Each item must be recognized by the
            `report_registry`. The iteration variable
            :math:`beta` will automatically be included.

        Notes
        -----
        deterministic:
            Rows initalized with a weight of 1 on the diagonal
            elements. This works out to be just the identity
            matrix and is the canonical starting point for DMQMC.

        random-uniform:
            Randomly selects diagonal determinants and adds
            a weight of 1 to that determinant. This can happen
            multiple times. This is how HANDE initializes the
            density matrix.

        fixed:
            Takes the optional parameter `fixed_diagonal` which is used as the
            diagonal of the density matrix.
        """
        super().setup(report_values)
        # Put iterator value at the front of the list
        #self._report_values = ["beta"] + self._report_values

        self._density_matrix = self._init_dm(initialization,
                                            n_particles,
                                            fixed_diagonal)

        self._S = np.zeros(self.system.n_determinants, dtype=np.float64)

    def _init_dm(self,
                 init: str,
                 particles: int,
                 diag: ArrayLike | None
                 ) -> Array:
        """
        CK Note: copied from functions.py::initialize_dm.

        I separated all methods with "thermal" in the name to an `IP_DMQMC`
        child class because based on the original docstring, those methods
        seemed to be designed for IP-DMQMC. Separating DMQMC and IP-DMQMC
        into different classes seemed like a conceptually useful thing to do.
        """
        if init == 'deterministic':
            randomrows = np.ones(self.system.n_determinants)

        elif init == 'random-uniform':
            randomrows = self._rng.choice(self.system.n_determinants,
                                          size=particles)
            randomrows = np.bincount(randomrows,
                                     minlength=self.system.n_determinants
                                     ).astype(np.float64)
        elif init == 'fixed':
            if len(diag) != self.system.n_determinants:
                raise RuntimeError(f"The length of 'diag' ({len(diag)}) "
                                   "must be equal to the number of "
                                   "determinants in the system.")
            randomrows = diag
        else:
            raise RuntimeError(f'Unknown initalization method {init}')

        f = np.diag(randomrows)
        return f

    def run(self,
            final_beta: float,
            dbeta: float,
            cycles_per_shift: int,
            shift_dampening: float,
            shift_by_rows: bool = False,
            spawn_cutoff: float = 0.01,
            n_add: float | None = None,
            ilevel: int | None = None,
            update_method: str = "euler",
            quiet: bool = False
            ):
        r"""
        Run a DMQMC realization.

        TODO: What are psips? Initiator & free level approximations?
        Comment on rounding below `|p_ij| > 1.0`

        Parameters
        ----------
        final_beta : float
            Target inverse temperature expressed as
            :math:`\beta = 1 / (k_\mathrm{B} T)`
        dbeta : float
            Size of a single update step in inverse temperature :math:`\beta`.
        cycles_per_shift : int
            Number of updates to :math:`\beta` made before updating
            the Hamiltonian shift.
        shift_dampening : float
            Affects how much the Hamiltonian shift varies as it updates
            every `cycles_per_shift` steps.
        shift_by_rows : bool, default false
            If True, calculate a shift for each row of the Hamiltonian.
            If False, calculate one shift for the entire Hamiltonian.
        spawn_cutoff : float, default 0.01
            Only accumulate psips if the change in a density matrix
            site :math:`|\partial p_{ik}| > \mathtt{spawn\_cutoff}`.
        n_add : float, default None
            If not `None`, utilize the initiator approximation
            and only allow spawning from sites :math:`p_{ij}` to empty
            sites :math:`p_{ik}` if :math:`|p_{ij}| > \mathtt{n_add}`.
        ilevel : int, default None
            If not `None`, utilize the initiator level approximation,
            allowing sites :math:`p_{ij}` to spawn if
            the difference in number of excitations between :math:`i`
            and :math:`j` is less than `ilevel`. Requires the system's
            `excitation_matrix` to be defineable
            if :math:`\texttt{ilevel} > 0`.
        update_method : str, default "euler"
            One of the supported update methods from (TODO link to)
            Iterative.parse_method()
        quiet : boolean, default False
            Silence printing the iteration report as the simulation runs.

        Notes
        -----
        The shift update follows Equation 16 in Blunt et al. 2014 [1]_.
        For more about the various approximations available, see
        :ref:`initiator-approximations`.

        References
        ----------
        .. [1] N. S. Blunt et al., "Density-matrix quantum Monte Carlo method,"
               Physical Review B, 89, 24, 2014
        """
        # Perform sanity checks
        if self._density_matrix is None:
            raise RuntimeError("You must first run the setup() method!")

        if ilevel is not None and not isinstance(ilevel, int):
            raise TypeError("Parameter ilevel must be type int; "
                            f"supplied value is type {type(ilevel)}.")

        # Run super()'s run method to ensure data safety.
        super().run()

        # While it makes sense for a parameter to be None when a feature
        # is disabled, Numba-compiled `propagate` methods in child classes
        # will require numeric values
        if n_add is None:
            n_add = 0.0
        if ilevel is None:
            ilevel = -1

        if ilevel > 0:
            self.system.generate_excitation_matrix()
            n_ex = self.system.excitation_matrix
            # Please remove this warning after the methods have been verified
            warnings.warn("Initiator level > 0 has not been "
                          "robustly verified. Please check for correctness "
                          "in all DMQMC child methods and remove this "
                          "warning.")
        else:
            # Make a dummy matrix with 0's on the diagonal
            # This allows classes w/ undefied generate_excitation_matrix()
            # to still work with ilevel 0. It also keeps Numba happy for
            # ilevel = None
            n_ex = np.ones((self.system.n_determinants,
                            self.system.n_determinants), dtype=np.int64) \
                 - np.eye(self.system.n_determinants)

        n_shifts = int(final_beta/(dbeta*cycles_per_shift))
        update_func = super().parse_method(update_method)
        rbr = 1 if shift_by_rows else None

        self._density_matrix = self._density_matrix

        # set initial shift
        # np will not be altered in this instance
        npsip = np.sum(self._density_matrix, axis=rbr)
        npsip = self._update_shift(self._density_matrix, npsip, cycles_per_shift,
                                   shift_dampening, dbeta, rbr)

        # Do initial reporting
        if not quiet:
            header = f"{'beta':>14}"
            for value in self._report_values:
                header += f" {value:>14}"
            print(header)
        self._do_report(0.0, quiet)

        for shift in range(n_shifts):

            for cycle in range(cycles_per_shift):

                self._density_matrix = update_func(self._propagate,  # f(dx/dy)
                                                   self._density_matrix,  # y
                                                   dbeta,  # stepsize dt
                                                   spawn_cutoff, n_add,  # args
                                                   ilevel, n_ex)  # args

                # Only store |p_ij| > 1.0, otherwise
                # round below this threshold in a non-biased manner
                # (stochastic rounding)
                replace = np.trunc(self._density_matrix +
                                   np.sign(self._density_matrix) *
                                   self._rng.random(self._density_matrix.shape))
                np.where(np.abs(self._density_matrix) < 1.0,
                         replace,
                         self._density_matrix)

            # update shift every report period
            npsip = self._update_shift(self._density_matrix,
                                       npsip, cycles_per_shift,
                                       shift_dampening, dbeta, rbr)

            # do periodic reporting
            self._do_report((shift+1)*cycles_per_shift*dbeta, quiet)

    def _do_report(self, current_beta: float, quiet: bool = False) -> None:
        """
        Put values for this current iteration into the self._data list.
        """
        current_data = {"beta": current_beta}
        rep_str = f"{current_beta:>14e}"
        for value in self._report_values:
            data = report_registry[value](self._density_matrix,
                                          **self._report_reqs[value])
            
            current_data[value] = data
            rep_str += f" {data:>14e}"

        if not quiet:
            print(rep_str)

        self._report_data.append(current_data) 

    def _update_shift(self,
                      p: Array,
                      np_old: Array,
                      A: int,
                      zeta: float,
                      dbeta: float,
                      rbr: int | None):
        npsip = np.abs(p).sum(axis=rbr)
        if rbr:
            for i in range(p.shape[0]):
                if npsip[i] != 0.0 and np_old[i] != 0.0:
                    self._S[i] -= (zeta/(A*dbeta))*np.log(npsip[i]/np_old[i])
        else:
            self._S -= (zeta/(A*dbeta))*np.log(npsip/np_old)

        return npsip

    def _propagate(self, p, *args, **kwargs) -> Array:
        """
        Wrap `_propagate_core` with the expected call signature.

        Numba-compiled functions do not have access to class attributes.
        Call signature is dictated by the "integrator" functions.
        """
        return self._propagate_core(p,
                                    self.system.hamiltonian,
                                    self._S,
                                    self._rng,
                                    *args,
                                    **kwargs)

    def _propagate_core(self, p, *args, **kwargs):
        raise NotImplementedError(
            "DensityMatrixQMC does not have it's own psip propagation "
            "method defined. Please use either SymmetricBlochDMQMC or "
            "AsymmetricBlochDMQMC, or a custom child class."
        )

    def save_data(self,
                  basename: str,
                  matrix_filetype: str = "csv",
                  report_filetype: str = "csv",
                  pickle_protocol: int | None = None) -> None:
        """
        Save the final density matrix and iteration report to file.

        The `basename` and `filetype` parameters will be used to construct
        filenames for all of the data written to file. For example, if
        `basename` is "test_run" and the `matrix_` and `report_filetype`
        are both "csv", the density matrix will be saved to
        "test_run_density_matrix.csv" and the iteration report will be saved to
        "test_run_report.csv".

        Parameters
        ----------
        basename : str
            Base name used to construct the filenames for the density
            matrix and iteration report
        matrix_filetype : str, default "csv"
            File type (aka extension) with which to save the density matrix.
            Supported types are:

            - "csv" : comma-separated value file
            - "npy" : NumPy binary file
            - "pkl" : Python pickle file
            - "txt" : text file (space-delimited)

        report_filetype : str, default "csv"
            File type (aka extension) with which to save the report.
            Supported types are:

            - "csv" : comma-separated value file
            - "txt" : text file (space-delimited)
            - "pkl" : pickle file

        pickle_protocol : unt, optional
        Protocol version to use if either `filetype` is "pkl".
        If none, uses `pickle`'s default.
        """
        save_array(self._density_matrix,
                   basename + "_density_matrix",
                   matrix_filetype,
                   pickle_protocol)
        save_report(self._report_data,
                    basename + "_report",
                    "beta",
                    report_filetype,
                    pickle_protocol)


class AsymmetricBlochDMQMC(DensityMatrixQMC):
    """
    Density matrix quantum Monte Carlo using the assymetric Bloch equation.

    TODO: write math here

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    rng_seed : int or array_like of ints, optional
        Seed or sequence of seeds for the psuedo random number generator.
        See :func:`numpy.random.default_rng`
    """

    def __init__(
            self,
            system: System,
            rng_seed: None | int | ArrayLike = None
            ) -> None:
        super().__init__(system, rng_seed)

    @staticmethod
    @njit
    def _propagate_core(p: Array,
                        H: Array,
                        S: Array,
                        rng,
                        cutoff: float,
                        nadd: float,
                        ilvl: int,
                        nex: Array):
        dets = p.shape[0]
        dp = np.zeros_like(p, dtype=np.float64)

        for i in range(dets):
            for j in range(dets):

                Stot = H[0, 0] + S[i]
                dp[i, j] = p[i, j] * \
                    (Stot - H[j, j])  # -(H_jj - S)

                p_ij = abs(p[i, j])

                # Iterate over sites that may spawn here at p_ij
                for k in range(dets):

                    if k == j:
                        continue

                    # While the docs write the rules as p_ij spawning at p_ik,
                    # we are actually checking if p_ik will
                    # spawn at/contribute to p_ij through the action of H_kj.

                    # The excitation matrix is not required for ilvl 0.
                    ichk = nex[i, k] <= ilvl

                    if abs(p[i, k]) > nadd or p_ij != 0.0 or ichk:
                        pr = p[i, k] * H[k, j]

                        if abs(pr) < cutoff:
                            pr /= cutoff
                            pr += np.sign(pr) * rng.random()
                            pr = np.trunc(pr)
                            pr *= cutoff

                        dp[i, j] -= pr  # -sum_k!=j(p_ik * H_kj)

        return dp


class SymmetricBlochDMQMC(DensityMatrixQMC):
    """
    Density matrix quantum Monte Carlo using the assymetric Bloch equation.

    TODO: write math here

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    rng_seed : int or array_like of ints, optional
        Seed or sequence of seeds for the psuedo random number generator.
        See :func:`numpy.random.default_rng`
    """

    def __init__(
            self,
            system: System,
            rng_seed: None | int | ArrayLike = None
            ) -> None:
        super().__init__(system, rng_seed)

    @staticmethod
    @njit
    def _propagate_core(p: Array,
                        H: Array,
                        S: Array,
                        rng,
                        cutoff: float,
                        nadd: float,
                        ilvl: int,
                        nex: Array):
        dets = p.shape[0]
        dp = np.zeros_like(p, dtype=np.float64)

        for i in range(dets):
            for j in range(dets):

                Stot = H[0, 0] + S[i]
                dp[i, j] = p[i, j]/2 * \
                    (Stot - H[i, i])
                dp[i, j] += p[i, j]/2 * \
                    (Stot - H[j, j])

                p_ij = abs(p[i, j])

                # Iterate over sites that may spawn here at p_ij
                for k in range(dets):

                    if k != j:
                        # While the docs write the rules as p_ij spawning at
                        # p_ik, we are actually checking if p_ik will
                        # spawn at/contribute to p_ij thru the action of H_kj.

                        ichk = nex[i, k] <= ilvl

                        if abs(p[i, k]) > nadd or p_ij != 0.0 or ichk:
                            pr = 0.5 * p[i, k] * H[k, j]

                            if abs(pr) < cutoff:
                                pr /= cutoff
                                pr += np.sign(pr) * rng.random()
                                pr = np.trunc(pr)
                                pr *= cutoff

                            dp[i, j] -= pr

                    if k != i:
                        # Now we check if p_kj can spwan at p_ij thru H_ik.

                        ichk = nex[k, j] <= ilvl

                        if abs(p[k, j]) >= nadd or p_ij != 0.0:
                            pr = 0.5 * H[i, k] * p[k, j]

                            if abs(pr) < cutoff:
                                pr /= cutoff
                                pr += np.sign(pr) * rng.random()
                                pr = np.trunc(pr)
                                pr *= cutoff

                            dp[i, j] -= pr

        return dp
