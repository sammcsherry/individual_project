
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt



def one_dimensional_chain():
    a = 0.24595  # [nm] unit cell length
    t = -2.8     # [eV] nearest neighbor hopping

    # Define a 1D lattice
    one_D_lattice = pb.Lattice(a1=[a, 0])
    one_D_lattice.add_sublattices(('A', [0, 0]))
    one_D_lattice.add_hoppings(
        ([1, 0], 'A', 'A', t)  # Hop from one 'A' site to the next
    )
    
    return one_D_lattice

lattice = one_dimensional_chain()
lattice.plot()
plt.show()
