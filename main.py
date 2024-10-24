from lattice_class import *
from evolve_wavefunction import *
import matplotlib.pyplot as plt
import numpy as np
from pre_set_structures import *


lattice2D = square_lattice(hopping_strength=2.8)
lattice2D.create_rectangle(width=11, height=11)
lattice2D.plot_model()
wavefunction2D = Wavefunction2D(lattice2D, wave_type='gaussian', time_step=0.1, num_steps=10, x0 = 5, y0 = 5, sigma=2)
wavefunction2D.create_gif()
