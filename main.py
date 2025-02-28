
from lattice_class import *
from evolve_wavefunction import *
import matplotlib.pyplot as plt
import numpy as np
from pre_set_structures import *
import matplotlib.pyplot as plt
from pybinding.repository import graphene



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
def main(charge):
    #unit_cell = graphene.bilayer()
    unit_cell = graphene.monolayer(nearest_neighbors=1, onsite=(0, 0))
    #unit_cell = square_lattice(-2.8)
    lattice = Lattice2D(unit_cell)
    width = 120
    height = 100
    lattice.create_model_2D(width, height)
    #lattice.plot()
    #lattice.add_periodic_boundary_x()
    lattice.add_pml(width_right=20, width_left=20, width_top=20, width_bottom=20, sigma=4, exponent=3, strength=0.7)
    lattice.add_coulomb_potential(x0=0, y0=0, charge=charge, epsilon=3.0)
    wavefunction2D = Wavefunction2D(lattice, wave_type='gaussian', time_step=5e-7, num_steps=10000, x0=-15, y0=0, sigma=10, kx=5, ky=0)
    wavefunction2D.evolve()
    wavefunction2D.create_gif()
    return wavefunction2D.transmission, wavefunction2D.reflection, wavefunction2D.angle_bin




def repeat_over_charge():
    transmission_history =[]
    charge_history = []
    with open('results.txt', 'w') as file:
        # Write a header to the file
        file.write("Charge\tTransmission\tReflection\tAngle_Bin\n")

        # Loop over charges
        for charge in np.arange(-50,50 , 10):
            # Run your main function
            transmission, reflection, angle_bin = main(charge)
            transmission_history.append(transmission)
            charge_history.append(charge)

            # Save the results to the file
            file.write(f"{charge:.2f}\t{transmission:.6f}\t{reflection:.6f}\t")
            np.savetxt(file, angle_bin[np.newaxis], fmt='%.6f', delimiter='\t')
        plt.scatter(charge, transmission)

import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from math import sqrt, pi


def bilayer_graphene():
    a = 0.24595  
    a_cc = 0.142 
    c = 0.335    
    t = -2.8      
    t_perp = -0.3  

    unit_cell = graphene.bilayer(onsite=[0,0,0,0.6])
    model = pb.Model(unit_cell, pb.translational_symmetry())
    #model.plot()
    plt.show()
    solver = pb.solver.lapack(model)
    a_cc = graphene.a_cc
    Gamma = [0, 0]
    K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
    M = [0, 2*pi / (3*a_cc)]
    K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

    bands = solver.calc_bands(K1, Gamma, M, K2)
    bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
    plt.show()



repeat_over_charge()


