"""Iterative DMQMC and IP-DMQMC methods."""

from .method import Iterative
from .. import systems

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
            system: systems.System,
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
              diag: ArrayLike | None = None,
              ) -> None:
        """
        Specify conditions for the DMQMC realization.

        These conditions include the initial density matrix and update method.

        CK Note: as it currently exists, this function could be folded into
        __init__ if a user should be forced to instantiate a new object
        when changing integration methods and/or the initial density matrix.

        Parameters
        ----------
        initialization : str, default "deterministic"
            Initialization method for the density matrix. See Notes for more.
            Must be one of:

            - deterministic
            - uniform-random
            - fixed

        n_particles : int, default 1
            The initial number of psip particles that should be present
            in the density matrix. Only used with the "deterministic" method.
        diag : array_like, optional
            Directly defined the diagonal of the density matrix when used
            with the "fixed" initialization method. The length of `diag`
            must be the same as the number of determinants in the system.

        Notes
        -----
        deterministic:
            Rows initalized with a weight of 1 on the diagonal
            elements. This works out to be just the identity
            matrix and is the canonical starting point for DMQMC.

        uniform-random:
            Randomly selects diagonal determinants and adds
            a weight of 1 to that determinant. This can happen
            multiple times. This is how HANDE initializes the
            density matrix.

        fixed:
            Takes the optional parameter `diag` which is used as the
            diagonal of the density matrix.
        """
        self._density_matrix = self._init_dm(initialization,
                                             n_particles,
                                             diag)
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

        elif init == 'uniform-random':
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
        spawn_cuttoff : float, default 0.01
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
        if self._density_matrix is None:
            raise RuntimeError("You must first run the setup() method!")

        if ilevel is not None and not isinstance(ilevel, int):
            raise TypeError("Parameter ilevel must be type int; "
                            f"supplied value is type {type(ilevel)}.")

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
            warnings.warn(f"Initiator level > 0 has not been "
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

        p = self._density_matrix

        # set initial shift
        # np will not be altered in this instance
        npsip = np.sum(p, axis=rbr)
        npsip = self._update_shift(p, npsip, cycles_per_shift,
                                   shift_dampening, dbeta, rbr)

        # print initial report
        print(f"{'Beta':>9}  {'Trace':>18}  "
              f"{'Energy':>18}  {'En/Tr':>18}")
        en = (p @ self.system.hamiltonian).trace()
        print(f"{0:>9.3f}  {p.trace():>18.12e}  "
              f"{en:>18.12e}  {en/p.trace():>18.12f}")

        for shift in range(n_shifts):

            for cycle in range(cycles_per_shift):

                p = update_func(self._propagate,  # func for dx/dy
                                p,      # y
                                dbeta,  # stepsize dt
                                spawn_cutoff, n_add,  # args for func
                                ilevel, n_ex)  # args for func

                # Only store |p_ij| > 1.0, otherwise
                # round below this threshold in a non-biased manner
                # (stochastic rounding)
                replace = np.trunc(p +
                                   np.sign(p)*self._rng.random(p.shape))
                np.where(np.abs(p) < 1.0,
                         replace,
                         p)

            # update shift every report period
            npsip = self._update_shift(p, npsip, cycles_per_shift,
                                       shift_dampening, dbeta, rbr)

            # do periodic reporting here
            en = (p @ self.system.hamiltonian).trace()
            print(f"{(shift+1)*cycles_per_shift*dbeta:>9.3f}  "
                  f"{p.trace():>18.12e}  "
                  f"{en:>18.12e}"
                  f"{en/p.trace():>18.12f}")

        # save final results
        self._density_matrix = p

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
            system: systems.System,
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
            system: systems.System,
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


class IPDensityMatrixQMC(DensityMatrixQMC):
    """
    Interaction-picture density matrix quantum Monte Carlo.

    Parameters
    ----------
    system : System object
        The predefined System to run the model with.
    """

    def __init__(
            self,
            system: systems.System,
            ) -> None:
        super().__init__(system)

    def setup(self,
              initialization: str = "deterministic",
              n_particles: int = 1,
              target_beta: float = 0.0,
              defined_thermal_weights: ArrayLike | None = None,
              ) -> None:
        """
        Set parameters for each of the realizations.

        Parameters
        ----------
        initialization : str, default "deterministic"
            Initialization method for the density matrix. See Notes for more.
            Must be one of:

            - deterministic-thermal
            - uniform-thermal
            - thermal-thermal
            - thermal-uniform

        n_particles : int, default 1
            TODO what does this mean
        target_beta : float, default 0.0
            Ignored if `defined_thermal_weights` is not `None`.
        defined_thermal_weights : array_like, optional
            Supply pre-defined thermal weights instead of using the
            auto-generated weights. Useful for, e.g., supplying FCI weights.

        Notes
        -----
        deterministic-thermal:
            Rows initialized with the thermal Hartree-Fock
            weights on the diagonal elements. The canonical
            starting point for IP-DMQMC.

        uniform-thermal:
            Uniformly selects random diagonal determinants and
            initalizes that row with a weight proportional to the
            thermal weight from FCI Hamiltonian.

        thermal-thermal:
            Selects random rows with a probability proportional
            to the thermal weight from the FCI Hamiltonian.
            This is not the correct way to initalize IP-DMQMC.

        thermal-uniform:
            Selects random rows based on probabilities proportional
            to the thermal weight of the FCI Hamiltonian diagonal
            elements. Then occupies that determinant with 1 walker.
        """
        self._init_dm(initialization,
                      n_particles,
                      target_beta,
                      defined_thermal_weights)

    def _init_dm(self,
                 init: str,
                 particles: int,
                 target: float,
                 thermal_weights: ArrayLike | None,
                 ):
        """
        CK Note: As noted in the docstring for DMQMC._init_dm,
        I made a separate `IP_DMQMC` class that supports different
        initializations for the density matrix, as the original
        functions.py::initialize_dm docstring implied these methods
        were suited for IP-DMQMC and it made conceptual sense
        to separate DMQMC and IP-DMQMC into separate classes.
        """
        if thermal_weights is not None:
            thermal_weights = thermal_weights
        elif 'thermal' in init:
            thermal_weights = np.exp(-target*np.diag(self.system.hamiltonian))
            thermal_weights /= thermal_weights.sum()

        if init == 'deterministic-thermal':
            randomrows = np.copy(thermal_weights)
        elif init == 'uniform-thermal':
            randomrows = self._rng.choice(self.system.n_determinants,
                                          size=particles)
            randomrows = np.bincount(randomrows,
                                     minlength=self.system.n_determinants
                                     ).astype(np.float64)
            randomrows *= thermal_weights
        elif init == 'thermal-thermal':
            randomrows = self._rng.choice(self.system.n_determinants,
                                          size=particles,
                                          p=thermal_weights)
            randomrows = np.bincount(randomrows,
                                     minlength=self.system.n_determinants
                                     ).astype(np.float64)
            randomrows *= thermal_weights
        elif init == 'thermal-uniform':
            randomrows = self._rng.choice(self.system.n_determinants,
                                          size=particles,
                                          p=thermal_weights)
            randomrows = np.bincount(randomrows,
                                     minlength=self.system.n_determinants
                                     ).astype(np.float64)
        else:
            print(' Unknown initalization method:', init)
            print(' Exiting...')
            return exit()

        f = np.diag(randomrows)
        occrows = np.count_nonzero(f)
        return f, occrows
