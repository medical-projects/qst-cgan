"""Functions and operations used throughout"""
import numpy as np
from qutip import displace
from qutip.random_objects import rand_dm


def add_state_noise(dm, sigma=0.01, sparsity=0.01):
    """
    Adds a random density matrices to the input state.
    
    .. math::
        \rho_{mixed} = \sigma \rho_0 + (1 - \sigma)\rho_{rand}$

    Args:
    ----
        dm (`qutip.Qobj`): Density matrix of the input pure state
        sigma (float): the mixing parameter specifying the pure state probability
        sparsity (float): the sparsity of the random density matrix
    
    Returns:
    -------
        rho (`qutip.Qobj`): the mixed state density matrix
    """
    hilbertsize = dm.shape[0]
    rho  = (1 - sigma)*dm + sigma*(rand_dm(hilbertsize, sparsity))
    rho = rho/rho.tr()
    return rho


def measure(alpha, rho=None):
    """
    Measures the photon number statistics for state rho when displaced
    by angle alpha.
    
    Parameters
    ----------    
    alpha: np.complex
        A complex displacement.
    Returns
    -------
    population: ndarray
        A 1D array for the probabilities for populations.
    """
    hilbertsize = rho.shape[0]
    D = displace(hilbertsize, -alpha)
    rho_disp = D*rho*D.dag()
    populations = np.real(np.diagonal(rho_disp.full()))
    return np.array(populations).reshape(-1)


def generalized_q(rho, xvec, yvec):
    hilbertsize = rho.shape[0]
    q = np.zeros(shape = (len(xvec), len(yvec), hilbertsize))
    for i, p in enumerate(yvec):
        for j, x in enumerate(xvec):
            beta = (x + 1j*p)/np.sqrt(2)
            q[i, j] = measure(beta, rho)
    return q
