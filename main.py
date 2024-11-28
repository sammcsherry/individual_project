from lattice_class import *
from evolve_wavefunction import *
import matplotlib.pyplot as plt
import numpy as np
from pre_set_structures import *
import matplotlib.pyplot as plt


lattice1D = chain_lattice(hopping_strength=1, chain_length= 1500)
lattice1D.create_model()
lattice1D.add_potential_gradient(potential=0.4, start_lattice=0, end_lattice= 1500)
lattice1D.add_gaussian_potential(U0 = -0.08, x0 = 550, sigma = 40)
wavefunction1D = Wavefunction1D(lattice=lattice1D,wave_type='gaussian', time_step=10, num_steps= 190, x0 = 250, sigma = 20)
#wavefunction1D.add_complex_absorbing_potential_exp(n = 100, potential_strength=0.1, decay_rate=7)
wavefunction1D.evolve()
wavefunction1D.momentum_space()
T, R = wavefunction1D.calculate_transmission()
wavefunction1D.create_gif()



#2D simulation
#lattice2D = graphene(hopping_strength=2.8)
#lattice2D.create_rectangle(width=60, height=50)
#lattice2D.plot_model()

#wavefunction2D = Wavefunction2D(lattice2D, wave_type='gaussian', time_step=0.3, num_steps=500, x0 = -4.0, y0 = 0, sigma=-1)
#lattice2D.add_complex_absorbing_potential_exp(potential_strength=1, decay_rate=1, n=5)
#lattice2D.get_hamiltonian()
#lattice2D.add_potential_gradient(slope=0.53)
#lattice2D.add_gaussian_potential(U0 = -4, x0 = 2.0, y0 = 0, sigma = -0.5)
#wavefunction2D.evolve()
#wavefunction2D.create_gif()