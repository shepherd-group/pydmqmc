#!/usr/bin/env bash

# For easier reproducibility.
export OMP_NUM_THREADS=1

# For more information about the flags used here, run
#python -u ./run_cla_psi.py
# Additionally one can find many flags not used here.

# Example calculations for three propagation methods:
#    1) RK(1) Euler's method
#    2) Symplectic
#    3) RK(2) midpoint
#    4) RK(2) Heun's method
#    5) RK(4)
# The examples simulate a transform-limited (TL) pulse applied
# to the system initially in the ground state.
# The simulation is run until 12.0 femtoseconds (-tf).
# The time step (-dt) is 1.0 attoseconds for deterministic evolution.
# The time step is 0.1 attoseconds for stochastic evolution.
# Statistics for the simulation are reported every 20 iterations (-n)
# for deterministic evolution, and every 200 for stochastic evolution
# to account for the smaller time step.

# Deterministic example calculations.
# Currently deterministic can only use a single core (-np 1),
# but threading is used/supported through numba.
mpiexec -n 1 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 1.0 \
    -n 20 \
    -m euler \
    -nc 1 \
    2>&1 | tee example_001A_euler.out

mpiexec -n 1 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 1.0 \
    -n 20 \
    -m symplectic \
    -nc 1 \
    2>&1 | tee example_002A_symplectic.out

mpiexec -n 1 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 1.0 \
    -n 20 \
    -m runge-kutta \
    -rkn 2 \
    -nc 1 \
    2>&1 | tee example_003A_rung_kutta_002_mp.out

mpiexec -n 1 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 1.0 \
    -n 20 \
    -m runge-kutta \
    -rkn 2 \
    -heun \
    -nc 1 \
    2>&1 | tee example_004A_rung_kutta_002_heun.out

mpiexec -n 1 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 1.0 \
    -n 20 \
    -m runge-kutta \
    -rkn 4 \
    -nc 1 \
    2>&1 | tee example_005A_rung_kutta_004.out

# Stochastic example calculations.
# Stochastic evolution supports MPI, and the example here uses 2 cores.
# Threading is no supported and is not used.
mpiexec -n 2 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 0.1 \
    -n 200 \
    -m euler \
    -nc 1 \
    -s \
    2>&1 | tee example_001B_stoch_euler.out

mpiexec -n 2 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 0.1 \
    -n 200 \
    -m symplectic \
    -nc 1 \
    -s \
    2>&1 | tee example_002B_stoch_symplectic.out

mpiexec -n 2 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 0.1 \
    -n 200 \
    -m runge-kutta \
    -rkn 2 \
    -nc 1 \
    -s \
    2>&1 | tee example_003B_stoch_rung_kutta_002_mp.out

mpiexec -n 2 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 0.1 \
    -n 200 \
    -m runge-kutta \
    -rkn 2 \
    -heun \
    -nc 1 \
    -s \
    2>&1 | tee example_004B_stoch_rung_kutta_002_heun.out

mpiexec -n 2 python -u ./run_cla_psi.py \
    -tf 12.0 \
    -dt 0.1 \
    -n 200 \
    -m runge-kutta \
    -rkn 4 \
    -nc 1 \
    -s \
    2>&1 | tee example_005B_stoch_rung_kutta_004.out
