from lattice_class import *
from evolve_wavefunction import *
import matplotlib.pyplot as plt
import numpy as np
from pre_set_structures import *
import matplotlib.pyplot as plt


unit_cell = chain_lattice(hopping_strength=1)
lattice = Lattice1D(unit_cell=unit_cell, width = 200)
#lattice.add_gaussion_potential(U0=100, x0=0, sigma=10)
wavefunction1D = Wavefunction1D(lattice=lattice,wave_type='gaussian', time_step=0.1, num_steps= 5, x0 = 0, sigma = 20, kx = np.pi/2+0.3)
wavefunction1D.evolve()
wavefunction1D.calculate_transmission_reflection()
wavefunction1D.create_gif()





#2D 
#unit_cell = graphene()
#lattice = Lattice2D(unit_cell)
#lattice.create_model_2D(width=200, height=200)
#wavefunction2D = Wavefunction2D(lattice, wave_type='gaussian', time_step=5, num_steps=5,  x0=0, y0=0, sigma=3, kx=1.5, ky=0)
#wavefunction2D.evolve()
#wavefunction2D.create_gif()


