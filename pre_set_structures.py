from lattice_class import *
import numpy as np

def graphene(hopping_strength):
    a_cc = 0.142
    a=a_cc*np.sqrt(3)
    a1 = [a, 0]
    a2 = [a/2, np.sqrt(3)*a/2]

    unit_vectors = [a1, a2]
    lattice2D = Lattice2D(unit_vectors)

    # Add sublattices A and B at different positions
    lattice2D.add_sublattice('A', [0, 0])
    lattice2D.add_sublattice('B', [0, a_cc])

    # Add nearest-neighbor hopping between sublattices A and B
    lattice2D.add_hopping([0,  0], 'A', 'B', hopping_strength)
    lattice2D.add_hopping([1,  -1], 'A', 'B', hopping_strength)
    lattice2D.add_hopping([0,  -1], 'A', 'B', hopping_strength)
    return lattice2D

def square_lattice(hopping_strength):
    a = 1.0
    a1 = [a, 0]
    a2 = [0, a]

    unit_vectors = [a1, a2]
    lattice2D = Lattice2D(unit_cell_length=1 , unit_vectors=unit_vectors)
    lattice2D.add_sublattice('A', [0, 0])
    lattice2D.add_hopping([1,  0], 'A', 'A', hopping_strength)
    lattice2D.add_hopping([0,  1], 'A', 'A', hopping_strength)

    return lattice2D

def chain_lattice(hopping_strength, chain_length):
    a = 1.0
    a1 = [a, 0]

    unit_vectors = a1
    lattice1D = Lattice1D(unit_vector= unit_vectors, chain_length=chain_length)
    lattice1D.add_sublattice('A', [0, 0])
    lattice1D.add_hopping([1, 0], 'A', 'A', hopping_strength)

    return lattice1D

