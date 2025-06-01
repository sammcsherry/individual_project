

from lattice_class import *
from evolve_wavefunction import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from scipy.interpolate import griddata


import matplotlib.pyplot as plt



def monolayer_graphene(charge, sigma, kx, charge_frequency):
    a_cc = 0.142  
    a = np.sqrt(3) * a_cc 
    height=600
    width=600
    K =  np.array([(4*pi)/(3*np.sqrt(3)*a_cc),0])
    print(f'K_x {K[0]}')
    print(f'K_y {K[1]}')
    unit_cell = graphene.monolayer(nearest_neighbors=1)
    lattice = Lattice2D(unit_cell = unit_cell, impurity='coulomb', height=height, width=width, x0=0, y0=16.614, beta = charge)
    #lattice.plot()
    #lattice.add_pml(width_right=50, width_left=50, width_top=50, width_bottom=50, sigma=5, exponent=3, strength=0.07)
    wavefunction2D = Wavefunction2D(lattice, wave_type='gaussian', time_step=1e-15, num_steps=250, charge_type='hole', charge_frequency=charge_frequency, x0=-100, y0=0, sigma=50, kx= K[0]+0.25, ky=K[1])

    #wavefunction2D.get_transfer_cross_section(width=width, height=height, charge=charge)
    wavefunction2D.evolve()
    wavefunction2D.get_transfer_cross_section(width=500, height=500, charge=charge)

    #wavefunction2D.get_transfer_cross_section(width=width, height=height)
    wavefunction2D.create_gif(f'wavefunction_gif_Q={charge}.gif')


def repeat_over_charge_frequncy():
    transmission_history =[]
    charge_history = []

    for charge_frequency in np.arange(2, 11, 1):
        transfer_cross_section= monolayer_graphene(charge=0.0, sigma=50, kx=0.25, charge_frequency=charge_frequency)
        with open('charge_frequency.txt', 'a') as file:
            file.write(f"{transfer_cross_section:.6f}\n\t")

def repeat_over_charge():
    for charge in np.arange(0.68, 0.8, 0.02):
        monolayer_graphene(charge=charge, sigma=50, kx=0.25, charge_frequency=0)

#repeat_over_charge()
#repeat_over_charge_frequncy()
