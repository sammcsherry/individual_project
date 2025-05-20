import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from pybinding.constants import hbar
from scipy.sparse.linalg import expm
from pybinding.repository import graphene
a = 0.24595   
a_cc = 0.142 
t = -2.8  
t_nn = 0.1  
vf = 3 / (2 * hbar) * abs(t) * a_cc  


class Lattice:
    def __init__(self, unit_cell, impurity):
        print(hbar)
        self.impurity = impurity
        self.unit_cell = unit_cell
        self.model = None
        self.hamiltonian = None
        self.hamiltonian_coo = None
        self.lattice_positions = None
        self.cross_section = []

    def plot(self, title="Lattice Model"):
        if self.model is None:
            raise ValueError("Model must be created before plotting.")
        self.model.plot()
        plt.title(title)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.axis("equal")
        plt.show()

class Lattice1D(Lattice):
    def __init__(self, unit_cell, width):
        super().__init__(unit_cell)
        self.width = width
        self.create_model_1D()
    
    def create_model_1D(self):

        self.model = pb.Model(self.unit_cell, pb.primitive(a1=self.width))
        pb.translational_symmetry(a1=True)
        self.hamiltonian = self.model.hamiltonian
        self.lattice_positions = self.model.system.positions

    def create_chain(self):
        length = self.chain_length * self.unit_cell_length
        shape = pb.line([0, 0], [length, 0]) 
        self.create_model_1D(shape)


    def add_linear_potential(self, slope):
        def linear_potential(positions):
            x_positions = positions[0]
            return slope * x_positions  

        self.add_potential(linear_potential)
    
    def add_gaussion_potential(self, U0, x0, sigma):
        potential = U0 * np.exp(-((self.lattice_positions[0] - x0) ** 2) / (2 * sigma ** 2))
        potential_matrix = csr_matrix(np.diag(potential))
        self.hamiltonian += potential_matrix

    def apply_periodic_boundary_conditions(self):
        lil_hamiltonian = self.hamiltonian.tolil()
        num_sites = lil_hamiltonian.shape[0]

        lil_hamiltonian[0, num_sites - 1] = -1  
        lil_hamiltonian[num_sites - 1, 0] = -1  
        self.hamiltonian = lil_hamiltonian.tocsr()

    def add_pml(self, width=2.0, sigma=1.0, exponent=1, strength = 0.1):
        x = self.lattice_positions[0]
        x_min, x_max = np.min(x), np.max(x)
    
        dx_left = np.maximum(0, width - (x - x_min))
        dx_right = np.maximum(0, width - (x_max - x))
        d = np.maximum(dx_left, dx_right)
        damping = (-1j * sigma * (d / width)**exponent)*strength

        """plt.figure()
        plt.plot(x, damping.real, label='Real part')
        plt.plot(x, damping.imag, label='Imaginary part')
        plt.xlabel("x")
        plt.ylabel("PML potential")
        plt.legend()
        #plt.show()#"""
        
        damping_mat = sp.diags(damping, format="csr")
        self.hamiltonian = self.hamiltonian + damping_mat



class Lattice2D(Lattice):
    def __init__(self, unit_cell, impurity, height, width,**kwargs):
        super().__init__(unit_cell, impurity)
        self.width = width
        self.height = height
        self.impurity = impurity
        self.create_model()
        self.potential_hamiltonian = csr_matrix(sp.diags(np.ones(len(self.lattice_positions[0]))))
        if impurity == 'coulomb':
            self.add_coulomb_potential(self.lattice_positions[0], self.lattice_positions[1], **kwargs)
        self.hamiltonian_coo = self.hamiltonian.tocoo()

    def create_model(self):
        shape = pb.rectangle(x=self.width, y=self.height)
        self.model = pb.Model(self.unit_cell, shape, graphene.constant_magnetic_field(magnitude=0.5))
        self.lattice_positions = self.model.system.positions
        self.hamiltonian = self.model.hamiltonian
        self.x = self.lattice_positions[0]
        self.y = self.lattice_positions[1]
        
    def add_gaussian_potential(self, U0, x0, y0, sigma):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]  
        potential = U0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        potential_matrix = csr_matrix(np.diag(potential))
        self.hamiltonian += potential_matrix

    def add_coulomb_potential(self, x,y,x0, y0, beta):
        scaled_beta = beta * hbar * vf
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        potential = scaled_beta / np.sqrt(r**2)
        self.potential_hamiltonian = csr_matrix(sp.diags(potential))
        print("lattice")
        self.hamiltonian += self.potential_hamiltonian

        #surface = plt.scatter(x, y, c=potential, cmap="viridis", s=40)

        #plt.colorbar(surface, shrink=0.5, aspect=5)
        #plt.show()


    
    def add_pml(self, 
            width_right=2.0, 
            width_left=2.0, 
            width_top=2.0, 
            width_bottom=2.0, 
            sigma=1.0, 
            exponent=3, 
            strength=0.1):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]
        
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        dx_right = np.maximum(0, width_right - (x_max - x))
        dx_left = np.maximum(0, width_left - (x - x_min))
        dy_top = np.maximum(0, width_top - (y_max - y))
        dy_bottom = np.maximum(0, width_bottom - (y - y_min))

        damping_right = (-1j * sigma * (dx_right / width_right)**exponent) * strength
        damping_left = (-1j * sigma * (dx_left / width_left)**exponent) * strength
        damping_top = (-1j * sigma * (dy_top / width_top)**exponent) * strength
        damping_bottom = (-1j * sigma * (dy_bottom / width_bottom)**exponent) * strength

        damping = damping_right + damping_left + damping_top + damping_bottom

        """plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=damping.imag, cmap='viridis', s=10)
        plt.colorbar(label='Imaginary Damping Strength')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("PML Damping Profile (Custom Widths for Each Edge)")
        plt.show()"""

        
        damping_mat = sp.diags(damping, format="csr")
        self.hamiltonian = self.hamiltonian + damping_mat

    def top_layer_potential(self, delta_V):
        x_positions = self.lattice_positions[0]
        energy = np.zeros(len(x_positions))
        mask = (self.model.system.sub == "A2") | (self.model.system.sub == "B2")
        energy[mask] = delta_V 
        return energy