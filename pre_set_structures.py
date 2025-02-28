from lattice_class import *
import numpy as np
def graphene():
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8      # [eV] nearest neighbor hopping

    lattice = pb.Lattice(a1=[a, 0], a2=[a/2, a/2 * np.sqrt(3)])
    lattice.add_sublattices(('A', [0, 0]), ('B', [0, a_cc]))

    # Standard hoppings
    lattice.add_hoppings(
        ([0,  0], 'A', 'B', t),
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lattice

def square_lattice(hopping_strength):
    a = 1.0
    a1 = [a, 0]
    a2 = [0, a]

    unit_vectors = [a1, a2]

    lattice = pb.Lattice(a1, a2)
    lattice.add_sublattices(('A', [0, 0]))
    lattice.add_hoppings(
        ([1,  0], 'A', 'A', hopping_strength),
        ([0,  1], 'A', 'A', hopping_strength)
    )
    return lattice

def hexagonal_lattice(hopping_strength):
    a = 1.0  
    sqrt3 = 3**0.5

    a1 = [a * 1.5, a * sqrt3 / 2]
    a2 = [a * 1.5, -a * sqrt3 / 2]


    lattice = pb.Lattice(a1, a2)
    lattice.add_sublattices(('A', [0, 0])) 
    lattice.add_hoppings(
        ([1, 0], 'A', 'A', hopping_strength),     
        ([0, 1], 'A', 'A', hopping_strength),       
        ([-1, 1], 'A', 'A', hopping_strength)      
    )

    return lattice



def chain_lattice(hopping_strength):
    a = 1.0
    t = hopping_strength 

    lattice = pb.Lattice(a1=[a]) 
    lattice.add_sublattices(('A', [0]))
    lattice.add_hoppings(
        ([1], 'A', 'A', t)
    )
    return lattice
