#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd

from typing import Any, Tuple
from numpy.typing import NDArray as Array

try:
    from integrals_readin import integral_system as readin
except ModuleNotFoundError:
    _pydmqmc_path = '/Users/vanbenschoten/Chemistry/codes/pydmqmc'
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(_script_dir, _pydmqmc_path))
    from integrals_readin import integral_system as readin


class Outer:
    def __init__(self, eigenvectors: Array) -> None:
        self.eigenvectors = eigenvectors

    def bra(self, state: int) -> Array:
        return self.eigenvectors[:, state]

    def ket(self, state: int) -> Array:
        return np.transpose(self.bra(state))

    def __getitem__(self, states: Tuple[int]) -> Array:
        return np.outer(self.ket(states[0]), self.bra(states[1]))


def get_empty_data_dict() -> dict:
    data = {
        'step': [],
        'energy_real': [],
        'energy_imag': [],
        'trace_real': [],
        'trace_imag': [],
        'pop1_real': [],
        'pop1_imag': [],
        'pop2_real': [],
        'pop2_imag': [],
        'pop_real': [],
        'pop_imag': [],
        'walkers': [],
    }
    return data


def get_coupled_hamiltonian(
        Ei: Array,
        outer: Any,
        state1: int,
        state2: int,
        driving: float,
    ) -> Array:

    E2 = Ei[state2]

    M22 = outer[state2, state2]
    M12 = outer[state1, state2]
    M21 = outer[state2, state1]

    H = E2 * M22 + driving * (M12 + M21)

    return H.astype(complex)


def report(
        step: int,
        rho: Array,
        H: Array,
        state1: int,
        state2: int,
        data: dict,
    ) -> dict:

    if step == 0:
        print(
            f'{"step":>10}  '
            f'{"real energy":>19}  '
            f'{"imag energy":>19}  '
            f'{"real trace":>19}  '
            f'{"imag trace":>19}  '
            f'{"real pop. 1":>19}  '
            f'{"imag pop. 1":>19}  '
            f'{"real pop. 2":>19}  '
            f'{"imag pop. 2":>19}  '
            f'{"real pop.":>18}  '
            f'{"imag pop.":>18}  '
            f'{"walkers":>18}'
        )

    trace = rho.trace()

    energy = (H @ rho).trace()/trace

    pop1 = rho[state1, state1]/trace
    pop2 = rho[state2, state2]/trace

    real = np.abs(rho.real).sum()
    imag = np.abs(rho.imag).sum()

    data['step'].append(step)
    data['energy_real'].append(energy.real)
    data['energy_imag'].append(energy.imag)
    data['trace_real'].append(trace.real)
    data['trace_imag'].append(trace.imag)
    data['pop1_real'].append(pop1.real)
    data['pop1_imag'].append(pop1.imag)
    data['pop2_real'].append(pop2.real)
    data['pop2_imag'].append(pop2.imag)
    data['pop_real'].append(real)
    data['pop_imag'].append(imag)
    data['walkers'].append(real + imag)

    print(
        f'{step:>10}  '
        f'{energy.real:> 16.12E}  '
        f'{energy.imag:> 16.12E}  '
        f'{trace.real:> 16.12E}  '
        f'{trace.imag:> 16.12E}  '
        f'{rho[state1, state1].real:> 16.12E}  '
        f'{rho[state1, state1].imag:> 16.12E}  '
        f'{rho[state2, state2].real:> 16.12E}  '
        f'{rho[state2, state2].imag:> 16.12E}  '
        f'{data["pop_real"][-1]:>16.12E}  '
        f'{data["pop_imag"][-1]:>16.12E}  '
        f'{data["walkers"][-1]:>16.12E}'
    )

    return data


def do_euler_integrator(
        tf: float,
        dt: float,
        H: Array,
        state1: int,
        state2: int,
        rho: Array = None,
        noise: Any = None,
    ) -> Any:
    # Propagate based on equations in:
    # https://doi.org/10.1063/1.5115323

    if rho is None:
        rho = np.zeros(H.shape, dtype=complex)
        rho[state1, state1] += 1.0

    data = get_empty_data_dict()
    data = report(0, rho, H, state1, state2, data)

    nsteps = int(tf/dt)

    for step in range(nsteps):
        Lrho = -1j*((H @ rho) - (rho @ H))

        if noise is not None:
            Lrho += (noise.random(Lrho.shape) - 0.5)/(2*Lrho.shape[0])
            Lrho += 1j*(noise.random(Lrho.shape) - 0.5)/(2*Lrho.shape[0])

        rho += dt*Lrho

        if (step + 1) % 100 == 0:
            data = report(step+1, rho, H, state1, state2, data)

    data = pd.DataFrame(data)

    data['tau'] = data['step']*dt

    return data


def do_symplectic_integrator(
        tf: float,
        dt: float,
        H: Array,
        state1: int,
        state2: int,
        rho: Array = None,
        noise: Any = None,
    ) -> Any:
    # Propagate based on equations in:
    # https://doi.org/10.1021/acs.jctc.8b00381

    if rho is None:
        rho = np.zeros(H.shape, dtype=complex)
        rho[state1, state1] += 1.0

    data = get_empty_data_dict()
    data = report(0, rho, H, state1, state2, data)

    nsteps = int(tf/dt)

    p = rho.imag
    q = rho.real

    for step in range(nsteps):
        dp = -((H @ q) - (q @ H))

        if noise is not None:
            dp += (noise.random(dp.shape) - 0.5)/(2*dp.shape[0])

        p = p + (0.5 if step == 0 else 1.0)*dt*dp

        dq = ((H @ p) - (p @ H))

        if noise is not None:
            dq += (noise.random(dq.shape) - 0.5)/(2*dp.shape[0])

        q = q + dt*dq

        if (step + 1) % 100 == 0:
            dp = -((H @ q) - (q @ H))
            rho = q + 1j*(p + 0.5*dt*dp)
            data = report(step+1, rho, H, state1, state2, data)

    data = pd.DataFrame(data)

    data['tau'] = data['step']*dt

    return data
