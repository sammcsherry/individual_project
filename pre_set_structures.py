from lattice_class import *
import numpy as np
def graphene():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping

    lattice = pb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2 * np.sqrt(3)])
    lattice.add_sublattices(('A', [0, 0]),
                        ('B', [0,  a_cc]))
    lattice.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lattice

def grapheneA():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbour hopping

    lattice = pb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2 * np.sqrt(3)])
    lattice.add_sublattices(('A', [0, 0]))
    return lattice




def square_lattice(hopping_strength):
    a = 1.0
    a1 = [a, 0]
    a2 = [0, a]

    unit_vectors = [a1, a2]

    lattice = pb.Lattice(a1, a2)
    lattice.add_sublattices(('A', [0, 0]))
    lattice.add_hoppings(
        # inside the main cell
        ([1,  0], 'A', 'A', hopping_strength),
        # between neighboring cells
        ([0,  1], 'A', 'A', hopping_strength)
    )
    return lattice

def hexagonal_lattice(hopping_strength):
    # Define lattice constant
    a = 1.0  # Nearest-neighbor distance
    sqrt3 = 3**0.5

    # Define unit cell vectors for the hexagonal lattice
    a1 = [a * 1.5, a * sqrt3 / 2]
    a2 = [a * 1.5, -a * sqrt3 / 2]

    # Create the lattice
    lattice = pb.Lattice(a1, a2)

    # Add a single site in the unit cell
    lattice.add_sublattices(('A', [0, 0]))  # One lattice site per unit cell

    # Define hoppings to nearest neighbors
    lattice.add_hoppings(
        ([1, 0], 'A', 'A', hopping_strength),       # Right neighbor
        ([0, 1], 'A', 'A', hopping_strength),       # Top-right neighbor
        ([-1, 1], 'A', 'A', hopping_strength)       # Top-left neighbor
    )

    return lattice



def chain_lattice(hopping_strength):
    a = 1.0  # [nm] unit cell length
    t = hopping_strength  # Hopping strength

    lattice = pb.Lattice(a1=[a])  # 1D lattice with one unit vector
    lattice.add_sublattices(('A', [0]))  # Single sublattice
    lattice.add_hoppings(
        # Hopping between neighboring cells
        ([1], 'A', 'A', t)
    )
    return lattice

