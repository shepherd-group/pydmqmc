"""Iterative IP-DMQMC method."""

from .dmqmc import DensityMatrixQMC
from ..systems import System

import numpy as np
from numba import njit
from scipy.optimize import newton

import warnings
from numpy.typing import ArrayLike, NDArray as Array


class InteractionPictureDMQMC(DensityMatrixQMC):
    """
    Interaction-picture density matrix quantum Monte Carlo.

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
        self._final_beta = None
        self._spawn_cutoff = None

    @property
    def final_beta(self) -> float | None:
        """Target inverse temperature."""
        return self._final_beta

    @property
    def spawn_cutoff(self) -> float | None:
        """Cutoff for psip spawning."""
        return self._spawn_cutoff

    def setup(self,
              final_beta: float,  # will also be used by self.run()
              initialization: str = "deterministic",
              n_particles: int = 1,
              spawn_cutoff: float = 0.01, # may also be used by self.run()
              defined_thermal_weights: ArrayLike | None = None,
              fixed_diagonal: ArrayLike | None = None
              ) -> None:
        r"""
        Set parameters for each of the realizations.

        Parameters
        ----------
        final_beta : float
            Target inverse temperature expressed as
            :math:`\beta = 1 / (k_\mathrm{B} T)`.
            Will also be used by the `run()` method for consistency.
            Ignored if `defined_thermal_weights` is not `None`.
        initialization : str, default "deterministic"
            Initialization method for the density matrix. See Notes for more.
            Must be one of:

            - deterministic
            - random-uniform
            - random-thermal
            - random-grand-canonical
            - fixed

        n_particles : int, default 1
            TODO what does this mean
        spawn_cutoff : float, default None
            Used with "random-grand-canonical" `initialization` method.
            If using this method, the `run()` method will inherit this value
            for consistency. In `run()`, this parameter will only
            allow psip spawns if the change in a density matrix
            site :math:`|\partial p_{ik}| > \mathtt{spawn\_cutoff}`.
            If the `initialization` method is anything other than
            "random-grand-canonical," this parameter is ignored.
        defined_thermal_weights : array_like, optional
            Supply pre-defined thermal weights instead of using the
            auto-generated weights. Useful for, e.g., supplying FCI weights.

        Notes
        -----
        deterministic:
            Rows initialized with the thermal Hartree-Fock
            weights on the diagonal elements. The canonical
            starting point for IP-DMQMC.

        random-uniform:
            Uniformly selects random diagonal determinants and
            initalizes that row with a weight proportional to the
            thermal weight from FCI Hamiltonian.

        random-thermal:
            Selects random rows based on probabilities proportional
            to the thermal weight of the FCI Hamiltonian diagonal
            elements. Then occupies that determinant with 1 walker.

        random-grand-canonical:
            TODO

        fixed:
            Takes the optional parameter `fixed_diagonal` which is used as the
            diagonal of the density matrix.
        """
        # Set values for use in run()
        self._final_beta = final_beta
        if initialization == "random-grand-canonical":
            self._spawn_cutoff = spawn_cutoff
        else:
            self._spawn_cutoff = None

        # Initialize density matrix
        self._density_matrix = self._init_dm(initialization,
                                             final_beta,
                                             n_particles,
                                             defined_thermal_weights,
                                             fixed_diagonal)
        self._S = np.zeros(self.system.n_determinants, dtype=np.float64)

    def _init_dm(self,
                 init: str,
                 target: float,
                 particles: int,
                 thermal_weights: ArrayLike | None,
                 diag: ArrayLike | None
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
        else:
            thermal_weights = np.exp(-target*np.diag(self.system.hamiltonian))
            thermal_weights /= thermal_weights.sum()

        if init == 'deterministic':
            randomrows = np.copy(thermal_weights)
        elif init == 'random-uniform':
            randomrows = self._rng.choice(self.system.n_determinants,
                                          size=particles)
            randomrows = np.bincount(randomrows,
                                     minlength=self.system.n_determinants
                                     ).astype(np.float64)
            randomrows *= thermal_weights
        elif init == 'random-thermal':
            randomrows = self._rng.choice(self.system.n_determinants,
                                          size=particles,
                                          p=thermal_weights)
            randomrows = np.bincount(randomrows,
                                     minlength=self.system.n_determinants
                                     ).astype(np.float64)
        elif init == 'fixed':
            if len(diag) != self.system.n_determinants:
                raise RuntimeError(f"The length of 'diag' ({len(diag)}) "
                                   "must be equal to the number of "
                                   "determinants in the system.")
            randomrows = diag
        elif init == "random-grand-canonical":
            randomrows = self._grand_canonical(particles)
        else:
            raise RuntimeError(f'Unknown initalization method {init}')

        f = np.diag(randomrows)
        return f
    
    def _grand_canonical(self, init_part):
        r"""
        Initialize to :math:`\exp(-\beta_T H^{(0)})`
        
        This is accomplised by sampling :math:`\exp(-\beta_T H^\prime)`,
        where :math:`H^\prime = \sum_{|D>} \epsilon_i`. We then
        re-weight based on the difference
        :math:`\exp(-\beta_T [H^{(0)} - H^\prime])`.
        """
        def __fermi_function(mu_ff, ei_ff, tb_ff):
            return 1.0/(np.exp(tb_ff*(ei_ff - mu_ff)) + 1.0)

        def __dnav_function(mu_df, ei_df, tb_df, nel_df):
            return nel_df - __fermi_function(mu_df, ei_df, tb_df).sum()

        # apply appropriate checks on these attributes
        eig = self.system.eigenvalues
        nel = self.system.n_electrons
        if eig is None or nel is None:
            raise RuntimeError("System must specify eigenvalues and "
                               "n_electrons in order to use the "
                               "random-grand-canonical initialization "
                               "method.")

        mu0 = eig[np.argsort(eig)][nel-1:nel+1].sum()/2.0
        mu = newton(__dnav_function, mu0, args=(eig, self._final_beta, nel), tol=1E-14)
        fi = __fermi_function(mu, eig, self._final_beta)
        print(f' \\beta_T = {self._final_beta:>18.12f}')
        print(f' \\mu = {mu:>18.12f}')

        if self.system.n_alpha is not None and self.system.n_beta is not None:
            nalpha = self.system.n_alpha
            nbeta = self.system.n_beta
        else:
            warnings.warn("System does not define both n_alpha and n_beta. "
                          "Assuming each are half of n_electrons "
                          "(with remainder randomly assigned).")
            nalpha = nel//2
            nbeta = nel//2
            if nel%2:  # randomly assign the remaider
                if self._rng.uniform() >= 0.5:
                    nalpha += 1
                else:
                    nbeta += 1
            assert nalpha + nbeta == nel

        # If needed, Hamiltonian is generated during super.__init__()
        hii = np.copy(np.diag(self.system.hamiltonian))
        eshift = hii[0] - eig[np.argsort(eig)][:nel].sum()
        bitints = self.system.get_bitarray_integers()

        rho0 = self._gci(self._rng,
                         self.system.bitarrays, 
                         init_part,
                         self.system.n_orbitals,
                         fi,
                         nalpha,
                         nbeta,
                         bitints,
                         hii,
                         eig,
                         self._final_beta,
                         eshift,
                         self._spawn_cutoff)

        return rho0

    @staticmethod
    @njit
    def _gci(rng,
             bars_gci: Array,
             initial_gci: int,
             norb_gci,
             fi_gci,
             nalpha_gci,
             nbeta_gci,
             bitints_gci,
             hii_gci,
             ei_gci,
             tb_gci,
             eshift_gci,
             cutoff_gci):
        nspawned = 0
        nel_gci = nalpha_gci + nbeta_gci
        rho0_gci = np.zeros(bars_gci.shape[0], dtype=np.float64)

        while nspawned < initial_gci:
            ba = np.zeros(norb_gci, dtype=np.int64)
            nsel, nsela, nselb, bitint = 0, 0, 0, 0

            for iorb in range(norb_gci):
                if rng.random() < fi_gci[iorb]:
                    occ = 1
                    ba[iorb] = occ
                    nsel += occ
                    nsela += int(occ*(iorb % 2 == 1))
                    nselb += int(occ*(iorb % 2))
                    bitint += int(occ*(2**iorb))

                if nsel > nel_gci or nsela > nalpha_gci or nselb > nbeta_gci:
                    allowed = False
                    bitint = -1
                    break
                else:
                    allowed = nsel == nel_gci

            if allowed and bitint in bitints_gci:
                energy = hii_gci[bitints_gci == bitint][0]
                energy -= ei_gci[ba == 1].sum()
                ps = np.exp(-tb_gci*(energy - eshift_gci))/cutoff_gci
                ps += rng.random()
                ps = np.trunc(ps)
                ps *= cutoff_gci

                rho0_gci[bitints_gci == bitint] += ps
                nspawned += int(ps)

        return np.trunc(rho0_gci + rng.random(rho0_gci.shape[0]))

    def run(self,
            dbeta,
            cycles_per_shift,
            shift_dampening,
            shift_by_rows = False,
            spawn_cutoff = 0.01,
            n_add = None,
            ilevel = None,
            update_method = "euler"):
        r"""
        Run an IP-DMQMC realization.

        For consistency, uses the value of `final_beta` set with
        `setup()`, even if `defined_thermal_weights` were supplied.

        TODO: What are psips? Initiator & free level approximations?
        Comment on rounding below `|p_ij| > 1.0`

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
            If the "random-grand-canonical" method was used for
            `initialization` during `setup()`, this parameter is ignored and
            the value of `spawn_cutoff` set during `setup()` is used instead.
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
        if self._spawn_cutoff is not None:  # did we set it during setup?
            spawn_cutoff = self._spawn_cutoff  # if yes, override argument

        return super().run(self._final_beta,
                           dbeta,
                           cycles_per_shift,
                           shift_dampening,
                           shift_by_rows,
                           spawn_cutoff,
                           n_add,
                           ilevel,
                           update_method)

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

                dp[i, j] = p[i, j] * \
                    (H[i,i] - H[j, j] + S[i])

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

                        dp[i, j] -= pr

        return dp