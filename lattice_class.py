import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix

from scipy.sparse import diags



class Lattice:
    def __init__(self, unit_cell_length):
        self.sites = 0
        self.unit_cell_length = unit_cell_length
        self.hamiltonian: csr_matrix
        self.sublattices = [] 
        self.hoppings = []
        self.model =None
        self.lattice_positions = None

    def add_sublattice(self, name, position):
        self.sublattices.append((name, position))

    def add_hopping(self, displacement, sublattice1, sublattice2, strength):
        self.hoppings.append((displacement, sublattice1, sublattice2, strength))

    def configure_lattice(self, lattice):
        # Common logic for both 1D and 2D lattices
        for name, position in self.sublattices:
            lattice.add_sublattices((name, position))
        for displacement, sub1, sub2, strength in self.hoppings:
            lattice.add_hoppings((displacement, sub1, sub2, strength))
        return lattice
    
    def plot_model(self):
        self.model.plot()
        plt.title("Lattice Model")
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.axis('equal')
        plt.show()

    def get_hamiltonian(self):
        return self.model.hamiltonian.todense()
        
    def set_hamiltonian(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def get_num_sites(self):
        return self.model.system.num_sites

    def get_hamiltonian_type(self):
        print(type(self.hamiltonian))

class Lattice1D(Lattice):
    def __init__(self, unit_vector, chain_length, unit_cell_length=1):
        super().__init__(unit_cell_length)
        self.num_sites = chain_length
        self.unit_vector = unit_vector

    def create_unit_cell(self):
        a1 = self.unit_vector
        lattice = pb.Lattice(a1=a1)  # 1D specific lattice creation
        return self.configure_lattice(lattice)

    def create_model(self):
        shape = self.chain_sites()
        lattice = self.create_unit_cell()
        self.model = pb.Model(lattice, shape)
        self.lattice_positions = self.model.system.positions
        self.hamiltonian = self.model.hamiltonian
        return self.model
    
    def chain_sites(self):
        chain_length = self.num_sites * self.unit_cell_length
        shape = pb.line([0, 0], [chain_length, 0]) 
        return shape
    
    def add_potential_gradient(self, potential, start_lattice, end_lattice):
        if self.model is None:
            raise ValueError("Model must be created before adding a potential gradient.")
        
        positions = self.model.system.positions 
        x_positions = positions[0]
        truncated_x_positions = x_positions[start_lattice:end_lattice] 
        potential_coeff = potential/len(truncated_x_positions)
        potential = np.zeros(len(x_positions))
        potential[start_lattice:end_lattice] = potential_coeff * truncated_x_positions
        potential_matrix = csr_matrix(np.diag(potential))
        self.hamiltonian += potential_matrix

        fig = plt.figure(figsize=(10, 8))
        plt.plot(x_positions, potential)
        plt.show()

    def add_gaussian_potential(self, U0, x0, sigma):
        positions = self.model.system.positions
        x_positions = positions[0]

        potential = U0 * np.exp(-((x_positions - x0)**2)/ (2 * sigma**2))

        potential_matrix = csr_matrix(np.diag(potential))

        self.hamiltonian += potential_matrix

        plt.plot(x_positions, potential)
        plt.show()

class Lattice2D(Lattice):
    def __init__(self, unit_vectors, unit_cell_length=0.24595):
        super().__init__(unit_cell_length)
        self.unit_vectors = unit_vectors


    def create_unit_cell(self):
        a1, a2 = self.unit_vectors[0], self.unit_vectors[1]
        lattice = pb.Lattice(a1=a1, a2=a2) 
        return self.configure_lattice(lattice)
    
    def create_rectangle(self, width, height):
        shape_obj = pb.rectangle(x=self.unit_cell_length * width,
                                     y=self.unit_cell_length * height)
        
        lattice = self.create_unit_cell()
        self.model = pb.Model(lattice, shape_obj)
        self.hamiltonian = self.model.hamiltonian

        self.lattice_positions = self.model.system.positions
        return self.model
    
    def add_potential_gradient(self, slope):
        if self.model is None:
            raise ValueError("Model must be created before adding a potential gradient.")
        
        positions = self.model.system.positions 
        x_positions = positions[0] 
        y_positions = positions[1]

        potential = slope * x_positions
        potential_matrix = csr_matrix(np.diag(potential))
        self.hamiltonian += potential_matrix

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_trisurf(x_positions, y_positions, potential, cmap="viridis", edgecolor='none')
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="Potential (U)")
        ax.set_title("2D Potential Difference (Scatter Surface Plot)")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        plt.show()

    def add_gaussian_potential(self, U0, x0, y0, sigma):
        positions = self.model.system.positions
        x_positions = positions[0]
        y_positions = positions[1]

        potential = U0 * np.exp(-((x_positions - x0)**2 + (y_positions - y0)**2) / (2 * sigma**2))

        potential_matrix = csr_matrix(np.diag(potential))

        self.hamiltonian += potential_matrix

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_trisurf(x_positions, y_positions, potential, cmap="viridis", edgecolor='none')
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="Potential (U)")
        ax.set_title("2D Gaussian Potential (Scatter Surface Plot)")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Potential (U)")
        plt.show()

    def add_complex_absorbing_potential_exp(self, potential_strength, n, decay_rate=1.0):
        num_sites = self.get_num_sites()

        complex_potential = np.zeros(num_sites, dtype=np.complex128)
        for i in range(n):
            left_scale = -1j * potential_strength * np.exp(-decay_rate * (i / n))
            right_scale = -1j * potential_strength * np.exp(-decay_rate * (i / n))

            complex_potential[i] = left_scale
            complex_potential[-(i + 1)] = right_scale

        diagonal_matrix = csr_matrix(np.diag(complex_potential))

        self.hamiltonian += diagonal_matrix
