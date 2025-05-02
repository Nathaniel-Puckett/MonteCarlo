"""
Unit and regression test for the montecarlo package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import montecarlo
import networkx
import numpy as np


def test_montecarlo_imported():
    
    assert "montecarlo" in sys.modules
    
def test_bitstring():
    
    bs = montecarlo.BitString(5)
    ref_bs = montecarlo.BitString(5)

    assert len(bs) == 5
    assert bs[0] == 0
    assert bs == ref_bs
    
    new_config = np.array([1, 0, 1, 1, 0])
    bs.set_config(new_config)
    assert (bs.config() == new_config).all()
    
    bs.set_integer_config(4)
    assert bs.integer() == 4
    
    bs.flip_site(2)
    assert bs.integer() == 0
    
    assert bs.on() == 0
    assert bs.off() == 5
    
def test_hamiltonian():
    
    pass