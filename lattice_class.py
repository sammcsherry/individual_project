import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import scipy.sparse as sp


class Lattice:
    def __init__(self, unit_cell):
        self.unit_cell = unit_cell
        self.model = None
        self.hamiltonian = None
        self.hamiltonian_coo = None
        self.lattice_positions = None
        self.cross_section = []

    def plot(self, title="Lattice Model"):
        """Plot the lattice."""
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
        """Create a PyBinding model with the configured lattice and shape."""
        self.model = pb.Model(self.unit_cell, pb.primitive(a1=self.width))
        pb.translational_symmetry(a1=True)
        self.hamiltonian = self.model.hamiltonian
        self.lattice_positions = self.model.system.positions

    def create_chain(self):
        """Create a 1D chain lattice."""
        length = self.chain_length * self.unit_cell_length
        shape = pb.line([0, 0], [length, 0]) 
        self.create_model_1D(shape)


    def add_linear_potential(self, slope):
        """Add a linear potential gradient along the chain."""
        def linear_potential(positions):
            x_positions = positions[0]
            return slope * x_positions  # Linear function of x

        self.add_potential(linear_potential)
    
    def add_gaussion_potential(self, U0, x0, sigma):
        potential = U0 * np.exp(-((self.lattice_positions[0] - x0) ** 2) / (2 * sigma ** 2))
        potential_matrix = csr_matrix(np.diag(potential))
        self.hamiltonian += potential_matrix

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
    def __init__(self, unit_cell):
        super().__init__(unit_cell)
    
    def create_model_2D(self, width, height):
        self.width = width
        self.height = height
        shape = pb.rectangle(x=width, y=height)
        self.model = pb.Model(self.unit_cell, shape)
        self.hamiltonian = self.model.hamiltonian
        self.hamiltonian_coo = self.hamiltonian.tocoo()
        self.lattice_positions = self.model.system.positions
        
    def add_gaussian_potential(self, U0, x0, y0, sigma):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]  
        potential = U0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        potential_matrix = csr_matrix(np.diag(potential))
        self.hamiltonian += potential_matrix

    def add_coulomb_potential(self, x0, y0, charge=1.0, epsilon=3.0, a = 0.3):

        distance = np.sqrt((self.lattice_positions[0] - x0)**2 + (self.lattice_positions[1] - y0)**2)
        distance = np.where(distance == 0, 1e-10, distance)
        potential = charge / (4 * np.pi * epsilon * np.sqrt(distance**2 + a**2))
        #plt.scatter(self.lattice_positions[0], self.lattice_positions[1], c = potential, cmap='viridis', s=4)
        #plt.colorbar(label='Imaginary Damping Strength')
        #plt.show()
        potential_matrix = csr_matrix(sp.diags(potential))
        self.hamiltonian += potential_matrix


    
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
        mask = (self.model.system.sub == "A2") | (x_positions.sub == "B2")
        energy[mask] = delta_V 
        return energy