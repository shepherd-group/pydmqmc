#!/usr/bin/env python

from .method import Iterative
from .. import systems

import numpy as np
from numba import njit

from numpy.typing import ArrayLike, NDArray as Array


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
              row_list: ArrayLike | None = None,
              diag: ArrayLike | None = None,
              ) -> None:
        """
        Specify conditions for the QMC realizations.

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
            - specific-rows
            - fixed

        n_particles : int, default 1
            TODO what does this mean
        defined_thermal_weights : array_like, optional
            Supply pre-defined thermal weights instead of using the
            auto-generated weights. Useful for, e.g., supplying FCI weights.
        row_list : array_like, optional
            Series of rows to be used with the "specific-rows"
            initialization method.
        diag : array_like, optional
            Directly defined the diagonal of the density matrix when used
            with the "fixed" initialization method.

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

        specific-rows:
            Takes the optional parameter `row_list` and occupies
            those specific rows with a weight of 1.

        fixed:
            Takes the optional parameter `diag` which is used as the
            diagonal of the density matrix.
        """
        self._density_matrix = self._init_dm(initialization,
                                             n_particles,
                                             row_list,
                                             diag)
        self._S = np.zeros(self.system.n_determinants, dtype=np.float64)

    def _init_dm(self,
                 init: str,
                 particles: int,
                 rows: ArrayLike | None,
                 diag: ArrayLike | None
                 ) -> Array:
        """
        CK Note: This is copied from functions.py::initialize_dm.
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
        elif init == 'specific-rows':
            randomrows = np.bincount(rows,
                                     minlength=self.system.n_determinants
                                     ).astype(np.float64)
        elif init == 'fixed':
            randomrows = diag
        else:
            raise RuntimeError(f'Unknown initalization method {init}')

        f = np.diag(randomrows)
        return f

    def run(self,
            final_beta: float,
            n_reports: int,
            n_cycles: int,
            spawn_cutoff: float = 0.01,
            n_add: int | None = None,
            ilevel: bool = False,
            flevel: bool = False,
            update_method: str = "euler",
            copy_dm: bool = False,
            ):
        r"""
        Run a DMQMC realization.

        TODO: What are psips? Initiator & free level approximations?
        Comment on rounding below |p_ij| > 1.0

        CK Note: instead of specifying tau, I thought it
        was more intuitive for the user to specify a number of reports.
        I'm not really sure what the initiator and free level
        approximations are.

        Parameters
        ----------
        final_beta : float
            Target temperature expressed as 
            :math:`\beta = 1 / (k_\mathrm{B} T)`
        n_reports : int
            Number of reports about system properties to produce
            as :math:`\beta` evolves towards `final_beta`.
        n_cycles : int
            Number of psip updates to perform per report.
        spawn_cuttoff : float, default 0.01
            Only accumulate psips if the change in a density matrix
            site :math:`|\partial p_{ik}| > \mathtt{spawn_cutoff}`.
        n_add : float, default None
            If not `None`, utilize the initiator approximation
            and limit psip spawning at denisty matrix site
            :math:`p_{ik}` to those :math:`|p_{ik}| \ge \mathtt{n_add}`.
        ilevel : bool, default False
            Turn on initiator level zero(?).
        flevel : bool, default False
            Free level; allow spawning to initiator level zero (from any site)
            regardless of its population.
        update_method : str, default "euler"
            One of the supported update methods from (TODO link to)
            Iterative.parse_method()
        copy_dm : bool, default False
            Evolve a copy of the density matrix created by the `setup` method.
            Though this consumes more memory, it can be useful for running
            multiple realizations.
        """
        if self._density_matrix is None:
            raise RuntimeError("You must first run the setup() method!")

        # While it makes sense for a parameter to be None when a feature
        # is disabled, Numba-compiled propagate methods in child classes
        # will require a float value
        if n_add is None:
            n_add = 0.0

        tau = final_beta / (n_reports * n_cycles)
        update_func = super().parse_method(update_method)

        if copy_dm:
            p = self._density_matrix.copy()
        else:
            p = self._density_matrix  # a mutable view

        # print initial report
        print(f"{'Beta':>9}  {'Trace':>18}  "
              f"{'Energy':>18}  {'En/Tr':>18}")
        en = (p @ self.system.hamiltonian).trace()
        print(f"{0:>9.3f}  {p.trace():>18.12e}  "
              f"{en:>18.12e}  {en/p.trace():>18.12f}")

        for report in range(n_reports):

            for cycle in range(n_cycles):

                p = update_func(self._propagate,  # func for dx/dy
                                p,    # x
                                None, # y
                                tau,  # stepsize dy
                                spawn_cutoff, n_add,  # args for func
                                ilevel, flevel  # args for func
                                )

                # Only store |p_ij| > 1.0, otherwise 
                # round below this threshold in a non-biased manner
                replace = np.trunc(p +
                                   np.sign(p)*self._rng.random(p.shape))
                np.where(np.abs(p) < 1.0,
                         replace,
                         p)

            # do periodic reporting here
            en = (p @ self.system.hamiltonian).trace()
            print(f"{(report+1)*n_cycles*tau:>9.3f}  {p.trace():>18.12e}  "
                  f"{en:>18.12e}  {en/p.trace():>18.12f}")

        # save final results
        self._density_matrix = p

    def _propagate(self, p, dummy, *args, **kwargs):
        raise NotImplementedError(
            "DensityMatrixQMC does not have it's own psip propagation "
            "method defined. Please use either SymmetricBlochDMQMC or "
            "AsymmetricBlochDMQMC."
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

    def _propagate(self, p, dummy, *args, **kwargs) -> Array:
        """
        This function matches the form expected by the integrators
        and is used to pass attributes into the Numba-compiled
        `_propagate_core` method.
        """
        return self._propagate_core(p, 
                                    self.system.hamiltonian,
                                    self._S,
                                    self._rng,
                                    *args,
                                    **kwargs)

    @staticmethod
    #@njit
    def _propagate_core(p: Array, 
                        H: Array, 
                        S: Array, 
                        rng,
                        cutoff: float,
                        nadd: float,
                        ilvl: bool,
                        flvl: bool):
        dets = p.shape[0]
        dp = np.zeros_like(p, dtype=np.float64)

        for i in range(dets):
            for j in range(dets):

                Stot = H[0,0] + S[i]
                dp[i,j] = p[i,j] * \
                    (Stot - H[j,j])  # -(H_jj - S)

                p_ij = abs(p[i,j])

                for k in range(dets):

                    if k == j:
                        continue

                    ichk1 = ilvl and i == k
                    ichk2 = flvl and i == j

                    if abs(p[i,k]) >= nadd or p_ij != 0.0 or ichk1 or ichk2:
                        pr = p[i,k] * H[k,j]

                        if abs(pr) < cutoff:
                            pr /= cutoff
                            pr += np.sign(pr) * rng.random()
                            pr = np.trunc(pr)
                            pr *= cutoff

                        dp[i,j] -= pr  # -sum_k!=j(p_ik * H_kj)

        return dp

class IPDensityMatrixQMC(DensityMatrixQMC):
    """
    Interaction-picture density matrix quantum Monte Carlo.
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
        # Construct data container
        data = {'Beta':[],
                'Shift':[],
                'Tr(Hp)':[],
                'Tr(p)':[],
                'Nw':[],
                '<E>':[],
                'N_rows':[]}
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