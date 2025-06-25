.. _methods-iterative:

Iterative Methods 
=================

Density-Matrix Quantum Monte Carlo
----------------------------------

.. _initiator-approximations:

initiator Approximations
++++++++++++++++++++++++

.. note::
    This text is Claire's best effort to understand these approximation methods 
    based on conversations with Will. Corrections are welcome.

The initiator rules are as follows.
For a given site :math:`\rho_{ij}` attempting to spawn a psip at :math:`\rho_{ik}`:

1. If :math:`\rho_{ik} \neq 0`, always let :math:`\rho_{ij}` spawn psips.
2. If :math:`\rho_{ik} = 0`, let :math:`\rho_{ij}` spawn if :math:`|\rho_{ij}| \ge n_\mathrm{add}`, represented by the parameter ``n_add``.
3. If neither of the above are satisfied but another :math:`\rho_{pq}`, where :math:`p \neq i` and :math:`q \neq j`, spawns to :math:`\rho_{ik}` with the *same sign*, then :math:`\rho_{ij}` may spawn as well.
4. If none of the above are satisfied, nullify the spawn from :math:`\rho_{ij}` to :math:`\rho_{ik}`.

.. warning::
    Claire isn't sure rule 3 is currently implemented in the code.

The **initiator approximation** is represented by step 2 and may be disabled by setting ``n_add = None`` (:math:`n_\mathrm{add} = 0`).
This allows all spawns from :math:`\rho_{ik}`.

The **initiator level approximation** modifies the above to include another rule before 4:
if the number of excitations between the determinant labels for :math:`i` and :math:`j` is
less than or equal to the supplied level, allow the spawn at :math:`\rho_{ik}`.
*Currently, only initiator level zero is supported.*
