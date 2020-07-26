"""
Tests for state generation code
"""
import pytest

import numpy as np

from qst_cgan.states import cat

from qutip import Qobj, coherent, fock


import pytest


def test_cat():
    """
	Tests the genreation of cat states

	TODO: add more meaningful tests for cats
	TODO: make a pytest.mark.parameterize to reduce redundancy
	"""
    cat1 = cat(8, 2)
    cat2 = cat(16, 2.5, S=1)

    assert cat1.shape == (8, 8)
    assert np.allclose(cat1.tr(), 1.0)

    assert cat2.shape == (16, 16)
    assert np.allclose(cat2.tr(), 1.0)

    assert cat1.isherm
    assert cat2.isherm


def test_fock():
    """
	Tests the genreation of cat states

	TODO: add more meaningful tests for cats
	TODO: make a pytest.mark.parameterize to reduce redundancy
	"""
    pass


def test_thermal():
    """
	Tests the genreation of thermal states
	"""
    pass
