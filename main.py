from lattice_class import *
from evolve_wavefunction import *
import matplotlib.pyplot as plt
import numpy as np
from pre_set_structures import *
import matplotlib.pyplot as plt

#unit_cell = chain_lattice(hopping_strength=1)
#lattice = Lattice1D(unit_cell=unit_cell, width = 200)
#lattice.plot()
#lattice.add_gaussion_potential(U0=100, x0=0, sigma=10)
#lattice.apply_periodic_boundary_conditions()
#lattice.add_pml( width=80, sigma=1, exponent=5, strength=0.7)
#wavefunction1D = Wavefunction1D(lattice=lattice,wave_type='gaussian', time_step=2, num_steps= 40, x0 = 0, sigma = 20, kx = -np.pi/2)
#wavefunction1D.evolve()
#wavefunction1D.create_gif()


#2D 
unit_cell = graphene()
lattice = Lattice2D(unit_cell)
width = 30
height = 30
lattice.create_model_2D(width, height)
#lattice.plot()
#lattice.add_periodic_boundary_x()
lattice.add_pml(width_right=8.0, width_left=5.0, width_top=5.0, width_bottom=5.0, sigma=2.0, exponent=3, strength=0.1)
lattice.add_coulomb_potential(x0=-2, y0=0, charge=2.5, epsilon=3.0)
wavefunction2D = Wavefunction2D(lattice, wave_type='gaussian', time_step=2, num_steps=140, x0=-10, y0=0, sigma=2, kx=1.3, ky=0)
wavefunction2D.evolve()
wavefunction2D.create_gif()


