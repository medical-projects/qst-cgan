"""
Generates various classes of states.
"""
import numpy as np

from qutip import Qobj
from qutip.states import fock_dm, thermal_dm, coherent_dm, coherent, basis, fock
from qutip.operators import displace
from qutip import Qobj
from qutip.random_objects import rand_dm

import random


from scipy.special import binom
from math import sqrt


def cat(N, alpha, S=None, mu=None):
    """
    Generates a cat state. For a detailed discussion on the definition
    see `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
    Args:
    -----
        N (int): Hilbert size dimension.
        alpha (complex64): Complex number determining the amplitude.
        S (int): An integer >= 0 determining the number of coherent states used
                 to generate the cat superposition. S = {0, 1, 2, ...}.
                 corresponds to {2, 4, 6, ...} coherent state superpositions.
        mu (int): An integer 0/1 which generates the logical 0/1 encoding of 
                  a computational state for the cat state.


    Returns:
    -------
        cat (:class:`qutip.Qobj`): Cat state density matrix
    """
    if S == None:
        S = 0

    if mu is None:
        mu = 0

    kend = 2 * S + 1
    cstates = 0 * (coherent(N, 0))

    for k in range(0, int((kend + 1) / 2)):
        sign = 1

        if k >= S:
            sign = (-1) ** int(mu > 0.5)

        prefactor = np.exp(1j * (np.pi / (S + 1)) * k)

        cstates += sign * coherent(N, prefactor * alpha * (-((1j) ** mu)))
        cstates += sign * coherent(N, -prefactor * alpha * (-((1j) ** mu)))

    rho = cstates * cstates.dag()
    return rho.unit()


def fock(N, n):
    """
    Returns the Fock state
    
    Args:
    -----
        N (int): Hilbert size dimension.
        n (int): Fock state to fill

    Returns:
    -------
        fock_dm (:class:`qutip.Qobj`): Fock state density matrix
    """
    return fock_dm(N, n)


def thermal(N, nth):
    """
    Returns the thermal  state
    
    Args:
    -----
        N (int): Hilbert size dimension.
        nth (float): Mean thermal photon number

    Returns:
    -------
        thermal_dm (:class:`qutip.Qobj`): Thermal state density matrix
    """
    return thermal_dm(N, nth)
