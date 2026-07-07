"""Iterative DMQMC methods using symmetric & asymmetric Bloch equations."""

from .method import Iterative
from ..systems import System
from ..utils import save_array, ParallelHelper

import numpy as np
from numba import njit

from numpy.typing import ArrayLike, NDArray as Array

import warnings  # remove when no longer in use


class DensityMatrixQMC(Iterative):
    """
    Density matrix quantum Monte Carlo.

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
        system: System,
        rng_seed: None | int | ArrayLike = None,
        parallel: bool = False,
    ) -> None:
        super().__init__(system, parallel)

        # Prepare the system, if needed.
        if self.system.hamiltonian is None:
            print("Generating Hamiltonian.")
            self.system.generate_hamiltonian()

        self._final_beta: float | None = None
        self._density_matrix: Array | None = None
        self._shift: Array | None = None

        if parallel:
            self._ph = ParallelHelper(
                shape=(self.system.n_determinants, self.system.n_determinants)
            )

        self.reset_rng(rng_seed)  # sets self._rng

    @property
    def density_matrix(self) -> None | Array:
        """Density matrix at current inverse temperature."""
        return self._density_matrix

    @property
    def final_beta(self) -> float | None:
        """Target inverse temperature."""
        return self._final_beta

    def setup(
        self,
        final_beta: float,
        initialization: str = "deterministic",
        n_particles: int = 1,
        fixed_diagonal: ArrayLike | None = None,
        report_quants: list[str] = ["trace", "energy expectation"],
    ) -> None:
        r"""
        Specify conditions for the DMQMC realization.

        This setup includes the initial density matrix and a data structure
        for reporting user-supplied quantities every iteration.

        Parameters
        ----------
        final_beta : float
            Target inverse temperature expressed as
            :math:`\beta = 1 / (k_\mathrm{B} T)`.
        initialization : str, default "deterministic"
            Initialization method for the density matrix. See Notes for more.
            Must be one of:

            - deterministic
            - random-uniform
            - fixed

        n_particles : int, default 1
            The initial number of psip particles that should be present
            in the density matrix. Only used with the "deterministic" method.
        fixed_diagonal : array_like, optional
            Directly defined the diagonal of the density matrix when used
            with the "fixed" initialization method. The length of `diag`
            must be the same as the number of determinants in the system.
        report_quants : list, optional
            List of quantities to periodically report while performing
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
        super().setup(report_quants)

        # Set values for use in run()
        # We set final_beta here to keep things consistent with IP-DMQMC,
        # which inherits this class. Some init methods in IP-DMQMC need to
        # know the final beta ahead of time.
        self._final_beta = final_beta

        if self._parallel:
            self._density_matrix = self._ph.safe_noncollective(
                self._init_dm, initialization, n_particles, fixed_diagonal
            )
        else:
            self._density_matrix = self._init_dm(
                initialization, n_particles, fixed_diagonal
            )

        self._shift = np.zeros(self.system.n_determinants, dtype=np.float64)

    def _init_dm(self, init: str, particles: int, diag: ArrayLike | None) -> Array:
        if init == "deterministic":
            randomrows = np.ones(self.system.n_determinants, dtype=np.float64)

        elif init == "random-uniform":
            randomrows = self._rng.choice(self.system.n_determinants, size=particles)
            randomrows = np.bincount(
                randomrows,
                minlength=self.system.n_determinants,
            )

        elif init == "fixed":
            if len(diag) != self.system.n_determinants:
                raise RuntimeError(
                    f"The length of 'diag' ({len(diag)}) "
                    "must be equal to the number of "
                    "determinants in the system."
                )
            randomrows = diag

        else:
            raise RuntimeError(f"Unknown initalization method {init}")

        f = np.diag(randomrows).astype(np.float64)
        return f

    def run(
        self,
        dbeta: float,
        cycles_per_shift: int,
        shift_dampening: float,
        shift_by_rows: bool = False,
        spawn_cutoff: float = 0.01,
        n_add: float | None = None,
        ilevel: int | None = None,
        integrator: str = "euler",
        quiet: bool = False,
    ):
        r"""
        Run a Density Matrix Quantum Monte Carlo realization.

        If a site (i,j) has a absolute psip weight less than one,
        it is stochastically rounded. The Hamiltonian is periodically shifted
        in order to stabilize the psip population.

        Parameters
        ----------
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
        integrator : str, default "euler"
            One of the supported integration methods from
            :meth:`pydmqmc.methods.Iterative.parse_integrator()`
        quiet : boolean, default False
            Silence printing the iteration report as the simulation runs.

        Notes
        -----
        The shift update follows Equation 16 in [1]_.
        For more about the various approximations available, see
        :ref:`initiator-approximations`.

        References
        ----------
        .. [1] Blunt, N. S., et al. (2014). Density-matrix quantum Monte Carlo method.
            Physical Review B, 89, 245124.
        """
        # Run super()'s run method to ensure data safety.
        super().run()

        # Perform sanity checks
        if self._density_matrix is None:
            raise RuntimeError("You must call setup() before run().")

        if ilevel is not None and not isinstance(ilevel, int):
            raise TypeError(
                "Parameter ilevel must be type int; "
                f"supplied value is type {type(ilevel)}."
            )

        if self._parallel:
            self._ph.allocate_reduce_buffers()
            start_index = self._ph.imin
            end_index = self._ph.imax
        else:
            start_index = 0
            end_index = self.system.n_determinants

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
            warnings.warn(
                "Initiator level > 0 has not been "
                "robustly verified. Please check for correctness "
                "in all DMQMC child methods and remove this "
                "warning."
            )
        else:
            # Make a dummy matrix with 0's on the diagonal
            # This allows classes w/ undefied generate_excitation_matrix()
            # to still work with ilevel 0. It also keeps Numba happy for
            # ilevel = None
            n_ex = np.ones(
                (self.system.n_determinants, self.system.n_determinants), dtype=np.int64
            ) - np.eye(self.system.n_determinants)

        n_shifts = int(self._final_beta / (dbeta * cycles_per_shift))
        integrator_func = super().parse_integrator(integrator)
        rbr = 1 if shift_by_rows else None

        # set initial shift
        # npsip will not be altered in this instance
        npsip = np.sum(self._density_matrix, axis=rbr)
        npsip = self._update_shift(
            self._density_matrix, npsip, cycles_per_shift, shift_dampening, dbeta, rbr
        )

        # Do initial reporting
        if self.is_reporter:
            if not quiet:
                header = f"{'beta':>14}"
                for value in self._report_quants:
                    header += f" {value:>14}"
                print(header)
        self.do_report("beta", 0.0, quiet)

        for shift in range(n_shifts):
            for cycle in range(cycles_per_shift):
                # Spawn psips
                psips = self._spawn(
                    dbeta,
                    self._density_matrix,
                    start=start_index,
                    end=end_index,
                    cutoff=spawn_cutoff,
                    nadd=n_add,
                    ilvl=ilevel,
                    nex=n_ex,
                )

                # Perform death/cloning
                self._density_matrix = integrator_func(
                    self._derivative,  # f(dy/dt)
                    self._density_matrix,  # y
                    dbeta,  # stepsize dt
                    self._ph,  # parallel helper (if applicable)
                    start=start_index,  # kwargs for _propagate_core
                    end=end_index,
                )

                # Perform annihilation (any oppositely signed psips will cancel)
                self._density_matrix += psips

                # Synchronize across processes a final time
                # (density matrix is also synchronized within the integration)
                if self._parallel:
                    self._ph.allreduce_sum(self._density_matrix)

            # update shift every report period
            npsip = self._update_shift(
                self._density_matrix,
                npsip,
                cycles_per_shift,
                shift_dampening,
                dbeta,
                rbr,
            )

            # do periodic reporting
            self.do_report("beta", (shift + 1) * cycles_per_shift * dbeta, quiet)

    def _spawn(self, dt: float, p: Array, *args, **kwargs) -> Array:
        """
        Wrap `_spawn_core`, a Numba-compiled function.

        Numba-compiled functions do not have access to class attributes.
        """
        return self._stochastic_round(
            self._spawn_core(dt, p, self.system.hamiltonian, self._rng, *args, **kwargs)
        )

    def _stochastic_round(self, matrix: Array) -> Array:
        # Only store |p_ij| > 1.0, otherwise
        # round below this threshold in a non-biased manner
        # (stochastic rounding)
        replace = np.trunc(matrix + np.sign(matrix) * self._rng.random(matrix.shape))
        np.where(np.abs(matrix) < 1.0, replace, matrix)
        return matrix

    def _spawn_core(
        self,
        dt: float,
        p: Array,
        hamiltonian: Array,
        rng: np.random.Generator,
        *args,
        **kwargs,
    ) -> Array:
        raise NotImplementedError(
            "DensityMatrixQMC does not have it's own psip spawning "
            "method defined. Please use either SymmetricBlochDMQMC or "
            "AsymmetricBlochDMQMC, or a custom child class."
        )

    def _derivative(self, p: Array, *args, **kwargs) -> Array:
        """
        Wrap `_propagate_core` with the expected call signature.

        Numba-compiled functions do not have access to class attributes.
        Call signature is dictated by the "integrator" functions.
        """
        return self._derivative_core(
            p, self.system.hamiltonian, self._shift, *args, **kwargs
        )

    def _derivative_core(
        self, p: Array, hamiltonian: Array, shift: Array, *args, **kwargs
    ) -> Array:
        raise NotImplementedError(
            "DensityMatrixQMC does not have it's own derivative "
            "method defined. Please use either SymmetricBlochDMQMC or "
            "AsymmetricBlochDMQMC, or a custom child class."
        )

    def _update_shift(
        self,
        p: Array,
        np_old: Array,
        A: int,
        zeta: float,
        dbeta: float,
        rbr: int | None,
    ):
        npsip = np.abs(p).sum(axis=rbr)
        if rbr:
            for i in range(p.shape[0]):
                if npsip[i] != 0.0 and np_old[i] != 0.0:
                    self._shift[i] -= (zeta / (A * dbeta)) * np.log(
                        npsip[i] / np_old[i]
                    )
        else:
            self._shift -= (zeta / (A * dbeta)) * np.log(npsip / np_old)

        return npsip

    def save_data(
        self,
        basename: str,
        matrix_filetype: str = "csv",
        report_filetype: str = "csv",
        pickle_protocol: int | None = None,
    ) -> None:
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
        if self.is_reporter:
            super().save_data(basename, "beta", report_filetype, pickle_protocol)
            save_array(
                self._density_matrix,
                basename + "_density_matrix",
                matrix_filetype,
                pickle_protocol,
            )


class AsymmetricBlochDMQMC(DensityMatrixQMC):
    r"""
    Density matrix quantum Monte Carlo using the assymetric Bloch equation.

    Density matrix quantum Monte Carlo propagates an ensemble of
    stochastic psi particles (psips). Each psip carries a weight and occupies a
    specific (i,j) site in the density matrix. During propagation,
    psips spawn, die, or change weight based on Hamiltonian matrix
    elements, implementing a Monte Carlo sampling of the density matrix
    evolution.

    In this forumulation, the density matrix starts at an inverse temperature
    :math:`\beta = 0` and is evolved towards the target :math:`\beta` according
    to the Bloch equation:

    .. math:: d\hat{\rho} / d\beta = -\hat{H} \hat{rho}

    where :math:`\hat{\rho}(\beta) = \exp(-\beta \hat{H})` is the unnormalized
    thermal density matrix and :math:`\hat{H}` is the Hamiltonian operator [1]_.

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    rng_seed : int or array_like of ints, optional
        Seed or sequence of seeds for the psuedo random number generator.
        See :func:`numpy.random.default_rng`
    parallel : bool, default False
        Whether to use MPI to parallelize the calculation.

    References
    ----------
    .. [1] Blunt, N. S., et al. (2014). Density-matrix quantum Monte Carlo method.
        Physical Review B, 89, 245124.
    """

    def __init__(
        self,
        system: System,
        rng_seed: None | int | ArrayLike = None,
        parallel: bool = False,
    ) -> None:
        super().__init__(system, rng_seed, parallel)

    @staticmethod
    @njit
    def _derivative_core(
        p: Array,
        H: Array,
        S: Array,
        start: int,
        end: int,
    ) -> Array:
        dets = p.shape[0]
        dp = np.zeros_like(p, dtype=np.float64)

        for i in range(start, end):  # only loop over assigned rows in parallel
            Stot = H[0, 0] + S[i]
            for j in range(dets):
                dp[i, j] = p[i, j] * (Stot - H[j, j])  # -(H_jj - S)

        return dp

    @staticmethod
    @njit
    def _spawn_core(
        dt: float,
        p: Array,
        H: Array,
        rng: np.random.Generator,
        start: int,
        end: int,
        cutoff: float,
        nadd: float,
        ilvl: int,
        nex: Array,
    ) -> Array:
        dets = p.shape[0]
        new_psips = np.zeros_like(p, dtype=np.float64)  # TODO make sparse

        for i in range(start, end):  # only loop over assigned rows in parallel
            for j in range(dets):
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
                        pr = -dt * p[i, k] * H[k, j]

                        if abs(pr) < cutoff:
                            pr /= cutoff
                            pr += np.sign(pr) * rng.random()
                            pr = np.trunc(pr)
                            pr *= cutoff

                        new_psips[i, j] += pr  # sum_k!=j(p_ik * H_kj)

        return new_psips


class SymmetricBlochDMQMC(DensityMatrixQMC):
    r"""
    Density matrix quantum Monte Carlo using the assymetric Bloch equation.

    Density matrix quantum Monte Carlo propagates an ensemble of
    stochastic psi particles (psips). Each psip carries a weight and occupies a
    specific (i,j) site in the density matrix. During propagation,
    psips spawn, die, or change weight based on Hamiltonian matrix
    elements, implementing a Monte Carlo sampling of the density matrix
    evolution.

    In this forumulation, the density matrix starts at an inverse temperature
    :math:`\beta = 0` and is evolved towards the target :math:`\beta` according
    to the symmetrized Bloch equation:

    .. math:: d\hat{\rho} / d\beta = -1/2(\hat{H} \hat{rho} + \hat{rho} \hat{H})

    where :math:`\hat{\rho}(\beta) = \exp(-\beta \hat{H})` is the unnormalized
    thermal density matrix and :math:`\hat{H}` is the Hamiltonian operator [1]_.

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    rng_seed : int or array_like of ints, optional
        Seed or sequence of seeds for the psuedo random number generator.
        See :func:`numpy.random.default_rng`
    parallel : bool, default False
        Whether to use MPI to parallelize the calculation.

    References
    ----------
    .. [1] Blunt, N. S., et al. (2014). Density-matrix quantum Monte Carlo method.
        Physical Review B, 89, 245124.
    """

    def __init__(
        self,
        system: System,
        rng_seed: None | int | ArrayLike = None,
        parallel: bool = False,
    ) -> None:
        super().__init__(system, rng_seed, parallel)

    @staticmethod
    @njit
    def _derivative_core(
        p: Array,
        H: Array,
        S: Array,
        start: int,
        end: int,
    ) -> Array:
        dets = p.shape[0]
        dp = np.zeros_like(p, dtype=np.float64)

        for i in range(start, end):  # only loop over assigned rows in parallel
            for j in range(dets):
                # Diagonal update (death/cloning)
                Stot = H[0, 0] + S[i]
                dp[i, j] = p[i, j] / 2 * (Stot - H[i, i])
                dp[i, j] += p[i, j] / 2 * (Stot - H[j, j])

        return dp

    @staticmethod
    @njit
    def _spawn_core(
        dt: float,
        p: Array,
        H: Array,
        rng: np.random.Generator,
        start: int,
        end: int,
        cutoff: float,
        nadd: float,
        ilvl: int,
        nex: Array,
    ) -> Array:
        dets = p.shape[0]
        new_psips = np.zeros_like(p, dtype=np.float64)  # TODO make sparse

        for i in range(start, end):  # only loop over assigned rows in parallel
            for j in range(dets):
                p_ij = abs(p[i, j])

                # Iterate over sites that may spawn here at p_ij
                for k in range(dets):
                    if k != j:
                        # While the docs write the rules as p_ij spawning at
                        # p_ik, we are actually checking if p_ik will
                        # spawn at/contribute to p_ij thru the action of H_kj.

                        ichk = nex[i, k] <= ilvl

                        if abs(p[i, k]) > nadd or p_ij != 0.0 or ichk:
                            pr = -dt * 0.5 * p[i, k] * H[k, j]

                            if abs(pr) < cutoff:
                                pr /= cutoff
                                pr += np.sign(pr) * rng.random()
                                pr = np.trunc(pr)
                                pr *= cutoff

                            new_psips[i, j] += pr

                    if k != i:
                        # Now we check if p_kj can spwan at p_ij thru H_ik.

                        ichk = nex[k, j] <= ilvl

                        if abs(p[k, j]) >= nadd or p_ij != 0.0:
                            pr = -dt * 0.5 * H[i, k] * p[k, j]

                            if abs(pr) < cutoff:
                                pr /= cutoff
                                pr += np.sign(pr) * rng.random()
                                pr = np.trunc(pr)
                                pr *= cutoff

                            new_psips[i, j] += pr

        return new_psips
