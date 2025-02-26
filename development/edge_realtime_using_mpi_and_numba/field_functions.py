#!/usr/bin/env python

import numpy as np

from scipy.special import erf
from numpy.typing import NDArray as Array


def cw_t(I: float, w: float, t: float | Array) -> float | Array:
    r''' Calculate the scaler function for a given time dependent continuous
    wave pulse given as
        E(t) = E_0 sin( w t ),
    where t is the time, E_0 is the field strength, and w is the carrier
    frequency. For more information see:
        http://dx.doi.org/10.1021/acs.jctc.8b00381

    Here we solve for E_0 using the field intensity, which is related like
        E_0 = \sqrt( \frac{2 I}{\mu_0 c} ),
    where I is the field intensity, \mu_0 is the vacuum permittivity, and c
    is the speed of light in a vacuum. Using Hartree atomic units the
    permittivity is simply 1/(4 pi), and the speed of light is given as
        c = 1 / \alpha,
    where \alpha is the fine structure constant. The Hartree atomic units
    for the intensity I should be
        I = \frac{E_h^2}{\hbar a_0^2}.

    Parameters
    ----------
    I : float
        The (average) field intensity in units of Hartree.
    w : float
        The field frequency in units of inverse time (1/(\hbar/E_h)).
    t : float | :class:`numpy.ndarray`
        The time in units of Hartree (\hbar/E_h).

    Returns
    -------
    Et : float | :class:`numpy.ndarray`
        The scaler field at the provided time(s) in units of Hartree.
    '''
    mu0 = 1.0/(4.0*np.pi)
    c = 1.0/0.0072973525643
    E0 = ( (2.0*I) / (mu0*c) )**0.5
    Et = E0 * np.sin(w * t)
    return Et


def tl_t(
        I: float,
        w: float,
        s: float,
        t0: float,
        t: float | Array,
    ) -> float | Array:
    r''' Calculate the scaler function for a given time dependent 
    transform-limited (TL) pulse given as
        E(t) = E_0 exp( -\frac{ (t - t_0)^2 }{ 2 s^2 } ) sin( w (t - t_0) )
    where t is the time, E_0 is the field strength, w is the carrier
    frequency, s is the pulse width, and t_0 is the pulse center.
    For more information see:
        http://dx.doi.org/10.1021/acs.jctc.8b00381

    For finding E_0 and unit conventions see `cw_t` above.

    Parameters
    ----------
    I : float
        The (average) field intensity in units of Hartree.
    w : float
        The field frequency in units of inverse time (1/(\hbar/E_h)).
    s : float
        The pulse width in units of time (\hbar/E_h).
    t0 : float
        The pulse center in units of time (\hbar/E_h).
    t : float | :class:`numpy.ndarray`
        The time in units of Hartree (\hbar/E_h).

    Returns
    -------
    Et : float | :class:`numpy.ndarray`
        The scaler field at the provided time(s) in units of Hartree.
    '''
    mu0 = 1.0/(4.0*np.pi)
    c = 1.0/0.0072973525643
    E0 = ( (2.0*I) / (mu0*c) )**0.5
    exp = np.exp( -( (t - t0)**2 )/( 2*(s**2) ) )
    Et = E0 * exp * np.sin(w * (t - t0))
    return Et


def c_t(
        I: float,
        w: float,
        s: float,
        t0: float,
        b: float,
        t: float | Array,
    ) -> float | Array:
    r''' Calculate the scaler function for a given time dependent
    chirped pulse given as
        E(t) = E_0 exp( -\frac{ (t - t_0)^2 }{ 2 s^2 } ) S(t)
    where
        S(t) = sin( (w + \frac{1}{2} b (t - t_0) ) (t - t_0) ),
    t is the time, E_0 is the field strength, w is the carrier frequency, s is
    the pulse width, b is the chirp parameter, and t_0 is the pulse center.
    For more information see:
        http://dx.doi.org/10.1021/acs.jctc.8b00381

    For finding E_0 and unit conventions see `cw_t` above.

    Parameters
    ----------
    I : float
        The (average) field intensity in units of Hartree.
    w : float
        The field frequency in units of inverse time (1/(\hbar/E_h)).
    s : float
        The pulse width in units of time (\hbar/E_h).
    t0 : float
        The pulse center in units of time (\hbar/E_h).
    b : float
        The chirp parameter in units of inverse time squared (\hbar/E_h)^{-2}.
    t : float | :class:`numpy.ndarray`
        The time in units of Hartree (\hbar/E_h).

    Returns
    -------
    Et : float | :class:`numpy.ndarray`
        The scaler field at the provided time(s) in units of Hartree.
    '''
    mu0 = 1.0/(4.0*np.pi)
    c = 1.0/0.0072973525643
    E0 = ( (2.0*I) / (mu0*c) )**0.5
    dt = t - t0
    exp = np.exp( -( (dt)**2 )/( 2*(s**2) ) )
    St = np.sin( ( w + (1/2) * b * (dt) ) * (dt) )
    Et = E0 * exp * St
    return Et
