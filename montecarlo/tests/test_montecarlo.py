"""
Unit and regression test for the montecarlo package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import montecarlo
import networkx as nx
import numpy as np


def test_montecarlo_imported():
    
    assert "montecarlo" in sys.modules
    
def test_bitstring():
    
    bs = montecarlo.BitString(5)
    ref_bs = montecarlo.BitString(5)

    assert bs == ref_bs
    assert bs[0] == 0
    assert len(bs) == 5
    assert str(bs) == "00000"
    
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
    
    G = nx.Graph()
    G.add_nodes_from([i for i in range(6)])
    G.add_edges_from([(i,(i+1)% G.number_of_nodes() ) for i in range(6)])
    for e in G.edges:
        G.edges[e]['weight'] = 2.0
        
    conf = montecarlo.BitString(6)
    ham = montecarlo.IsingHamiltonian(G)
    ham.set_mu(np.array([.1 for i in range(6)]))

    conf.flip_site(2)
    conf.flip_site(3)
    e = ham.energy(conf)
    assert(np.isclose(e, 3.8))
    
    E, M, HC, MS = ham.compute_average_values(1)
    
    assert(np.isclose(E,  -11.95991923))
    assert(np.isclose(M,   -0.00000000))
    assert(np.isclose(HC,   0.31925472))
    assert(np.isclose(MS,   0.01202961))

def test_montecarlo():
    
    G = nx.Graph()
    G.add_nodes_from([i for i in range(6)])
    G.add_edges_from([(i,(i+1)% G.number_of_nodes() ) for i in range(6)])
    for e in G.edges:
        G.edges[e]['weight'] = 2.0
        
    conf = montecarlo.BitString(6)
    ham = montecarlo.IsingHamiltonian(G)
    ham.set_mu(np.array([.1 for i in range(6)]))
    
    T = 2
    
    Eref, Mref, HCref, MSref = ham.compute_average_values(T)

    mc = montecarlo.MonteCarlo(ham)
    E, M = mc.run(T=T, n_samples=5000, n_burn=100)

    Eavg = np.mean(E)
    Estd = np.std(E)
    Mavg = np.mean(M)
    Mstd = np.std(M)

    HC = (Estd**2)/(T**2)
    MS = (Mstd**2)/T