import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import copy as cp
from collections import deque

class BitString:
    """
    Simple class to implement a config of bits.
    """
    
    def __init__(self, bitstring_length):
        """
        Initializes the bitstring as a constant array of zeros based on bitstring_length.
        
        Parameters
        ----------
        bitstring_length : int
            Specifies the number of desired bits to be created.
        
        Returns
        -------
        None
        """
        self.bits = np.zeros(bitstring_length, dtype = np.int64)

    def __str__(self):
        """
        Returns the bitstring as a string.
        
        Parameters
        ----------
        None

        Returns
        -------
        bit_string : str
            A string representation of the binary sequence.
        """
        bit_string = ""
        for bit in self.bits:
            bit_string += f"{bit}"
        return bit_string

    def __len__(self):
        """
        Returns the length of the bitstring.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        len(self.bits) : int
            Length of the sequence of bits.
        """
        return len(self.bits)

    def __eq__(self, other):
        """
        Compares two bitstrings to each other.
        
        Parameters
        ----------
        other : BitString
            The bitstring being compared to.
        
        Returns
        -------
        self.bits == other.bits : boolean
            Equality of bitstrings.
        """
        return (self.bits == other.bits).all()
    
    def __getitem__(self, a):
        return self.bits[a]

    def config(self):
        """
        Returns the bitstring as a list of bits.
        
        Parameters
        ----------
        None

        Returns
        -------
        self.bits : list
            The sequence of bits in list format.
        """
        return self.bits

    def set_config(self, array):
        """
        Sets the bitstring to a new sequence.
        
        Parameters
        ----------
        array : np.array
            The new bit sequence.
        
        Returns
        -------
        None
        """
        self.bits = np.array(array, dtype = np.int64)

    def integer(self):
        """
        Gets the integer value of the bitstring.
        
        Parameters
        ----------
        None

        Returns
        -------
        bin_to_int(self.bits) : int
            Function called that converts binary to integer.
        """
        return bin_to_int(self.bits)

    def set_integer_config(self, num):
        """
        Sets the bitstring to a new sequence equal to an integer.
        
        Parameters
        ----------
        num : int
            The integer to change the value of the bitstring to.
        
        Returns
        -------
        None
        """
        int_to_bin(num, self.bits)

    def flip_site(self, bit):
        """
        Gets a target bit and flips its value.
        
        Parameters
        ----------
        bit : int
            The location of the bit to flip.
        
        Returns
        -------
        None
        """
        target = self.bits[bit]
        if target == 1:
            self.bits[bit] = 0
        elif target == 0:
            self.bits[bit] = 1

    def on(self):
        """
        Determines how many bits are 1 (on).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bit_list.count(1) : list
            Array converted to list counts the number of ones.
        """
        bit_list = list(self.bits)
        return bit_list.count(1)

    def off(self):
        """
        Determines how many bits are 0 (off).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bit_list.count(0) : list
            Array converted to list counts the number of zeroes.
        """
        bit_list = list(self.bits)
        return bit_list.count(0)

class IsingHamiltonian:
    
    def __init__(self, G):
        """
        Initializes the internal networkx graph and bitstring.
        
        Parameters
        ----------
        G : networkx graph
            A networkx graph representing particle interactions.
        
        Returns
        -------
        None
        """
        self.G = G
        self.bs = BitString(len(self.G))
        self.weights = list()
        for edge in self.G.edges:
            self.weights.append(self.G.edges[edge]["weight"])
        self.mu = list(np.full(len(self.G), 0))
        self.N = len(self.G)
    
    def set_mu(self, mu):
        """
        Defines the internal mu values, which are indicitive of a magnetic field.
        
        Parameters
        ----------
        mu : np.array
            An array of mu values.
        
        Returns
        -------
        None
        """
        self.mu = mu
        self.N = len(mu)
        
    def energy(self, bs):
        """
        Calculates the energy of a given bitstring by considering the weights of G.
        
        Parameters
        ----------
        bs : BitString
            The bitstring to calculate the energy of.
        
        Returns
        -------
        total_energy : float
            The energy of a given bitstring.
        """
        connection_matrix = nx.adjacency_matrix(self.G).todense()
        total_energy = 0
            
        for i in range(len(connection_matrix)):
            target_spin = bit_to_spin(bs[i])
            for j in range(i, len(connection_matrix)):
                if connection_matrix[i][j]:
                    connected_spin = bit_to_spin(bs[j])
                    total_energy += self.weights[i+j-1] * target_spin * connected_spin
            total_energy -= self.mu[i] * target_spin
        
        return total_energy
    
    def magnetization(self, bs):
        """
        Calculates the magnetization of a given bitstring by adding up spins.
        
        Parameters
        ----------
        bs : BitString
            The bitstring to calculate the magnetization of.
        
        Returns
        -------
        total_magnetization : float
            The magnetization of a given bitstring.
        """
        total_magnetization = 0
        for i in bs.config():
            total_magnetization += bit_to_spin(i)
        return total_magnetization
    
    def compute_average_values(self, T:float):
        """
        Calculates four average values that are thermodynamically relevant.
        
        Parameters
        ----------
        T : float
            The temperature to use for calculations
        
        Returns
        -------
        E : float
            The average energy
        M : float
            The average magnetization
        HC : float
            The heat capacity
        MS : float
            The magnetic susceptibility
        """
        P = probability(self.bs, self.G, T)
        E = average_energy(self.bs, self.G, P, 1)
        EE = average_energy(self.bs, self.G, P, 2)
        M = average_magnetization(self.bs, P, 1)
        MM = average_magnetization(self.bs, P, 2)
        
        HC = (EE - E**2) * (T**-2)
        MS = (MM - M**2) * (T**-1)
        
        return E, M, HC, MS
    
class MonteCarlo:
    
    def __init__(self, ising_hamiltonian):
        """
        Initializes with a given ising hamiltonian
        
        Parameters
        ----------
        ising_hamiltonian : IsingHamiltonian
            A IsingHamiltonian object used for calculating values
        
        Returns
        -------
        None
        """
        self.ih = ising_hamiltonian
        
    def run(self, T, n_samples, n_burn):
        """
        Runs the metropolis sampling using the given parameters
        
        Parameters
        ----------
        T : float
            The temperature of the system
        n_samples : int
            Number of times to compute values
        n_burn : int
            Number of values to trim from the front of the list
        
        Returns
        -------
        energies : list
            A list of computed energies
        magnetizations : list
            A list of computed magnetizations
        """
        bs_i = self.ih.bs
        energy_list = list()
        magnet_list = list()
        
        for i in range(n_samples):
            for n in range(len(bs_i)):
                bs_j = cp.deepcopy(bs_i)
                bs_j.flip_site(n)
                bs_i = new_configuration(self.ih, bs_i, bs_j, T)
            energy_list.append(self.ih.energy(bs_i))
            magnet_list.append(self.ih.magnetization(bs_i))
        
        energies = energy_list[n_burn:]
        magnetizations = magnet_list[n_burn:]
            
        return energies, magnetizations

def int_to_bin(num, bn_list):
    for i in range(1, len(bn_list)+1):
        num, rem = divmod(num, 2)
        bn_list[-i] = rem

def bin_to_int(bn_list):
    num = 0
    for i in range(len(bn_list)):
        num += bn_list[-i-1] * (2**i)
    return num

def bit_to_spin(bit):
    return 1 if bit == 0 else -1 if bit == 1 else None

def energy(bs: BitString, G: nx.Graph):
    """
    Gets the energy of a given bitstring in relation to connections on a graph
    Parameters:
    - bs : Input bitstring, will determine spin of each node on G
    - G : Graph with connected nodes indicating interactions
    Returns:
    - total_energy : The energy of a given state in graph G
    """
    connection_matrix = nx.adjacency_matrix(G).todense()
    weights = list()
    total_energy = 0
        
    for edge in G.edges:
        weights.append(G.edges[edge]["weight"])
        
    for i in range(len(connection_matrix)):
        target_spin = bit_to_spin(bs[i])
        for j in range(i, len(connection_matrix)):
            if connection_matrix[i][j]:
                connected_spin = bit_to_spin(bs[j])
                total_energy += weights[i] * target_spin * connected_spin
    
    return total_energy

def probability(bs, G, T):
    
    individual_probabilites = list()
    Z = 0
    beta = (T)**-1
    max_value = 2**len(bs)
    
    for i in range(max_value):
        bs.set_integer_config(i)
        i_energy = energy(bs, G)
        P_i = np.exp(-beta * i_energy)
        individual_probabilites.append(P_i)
        Z += P_i
        
    return individual_probabilites / Z

def average_magnetization(bs, P_alpha, factor):
    
    AM = 0
    max_value = 2**len(bs)
    
    for i in range(max_value):
        M = 0
        bs.set_integer_config(i)
        for j in bs.config():
            M += bit_to_spin(j)
        AM += (M**factor) * P_alpha[i]
            
    return AM

def average_energy(bs, G, P_alpha, factor):
    
    AE = 0
    max_value = 2**len(bs)
    
    for i in range(max_value):
        bs.set_integer_config(i)
        AE += (energy(bs, G)**factor) * P_alpha[i]
        
    return AE

def new_configuration(ih, bs_i, bs_j, T):
    e_i = ih.energy(bs_i)
    e_j = ih.energy(bs_j)
    
    if e_i >= e_j:
        return bs_j
    else:
        Ws = np.exp((e_i - e_j)/T)
        if Ws >= random.random():
            return bs_j
        else:
            return bs_i

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    pass

