#!/usr/bin/env python

from dataclasses import dataclass

@dataclass(
    init=True,
    repr=True,
    frozen=True,
)
class _units:
    # W / a.u.
    W: float = 1.80237834206756E-01
    # cm / a.u.
    cm: float = 5.29177210544E-9
    # fs / a.u.
    fs: float = 2.4188843265864E-02
    # as / a.u.
    ats: float = 2.4188843265864E+01

units = _units()
