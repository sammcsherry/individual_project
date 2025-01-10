import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


class Lattice:
    def __init__(self, unit_cell):
        self.unit_cell = unit_cell
        self.model = None
        self.hamiltonian = None
        self.lattice_positions = None

    def add_gaussion_potential(self, potential_function):
        """Apply a potential to the lattice based on a custom function."""
        if self.model is None:
            raise ValueError("Model must be created before adding potentials.")
        positions = self.model.system.positions
        potential = potential_function(positions)
        potential_matrix = csr_matrix(np.diag(potential))
        self.hamiltonian += potential_matrix

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
        self.hamiltonian = self.model.hamiltonian
        self.lattice_positions = self.model.system.positions

    def create_chain(self):
        """Create a 1D chain lattice."""
        length = self.chain_length * self.unit_cell_length
        shape = pb.line([0, 0], [length, 0])  # A straight line in 1D
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


class Lattice2D(Lattice):
    def __init__(self, unit_cell):
        super().__init__(unit_cell)

    
    def create_model_2D(self, width, height):
        """Create a PyBinding model with the configured lattice and shape."""
        self.width = width
        self.height = height
        shape = pb.rectangle(x=width, y=height)
        self.model = pb.Model(self.unit_cell, pb.primitive(a1=width, a2=height))
        self.hamiltonian = self.model.hamiltonian
        self.lattice_positions = self.model.system.positions
        

    def create_rectangle(self, width, height):
        # Use the pb.rectangle shape to define the region
        shape = pb.rectangle(width, height)

        self.model = pb.Model(self.unit_cell, pb.primitive(a1=width, a2=height))
        self.hamiltonian = self.model.hamiltonian
        self.lattice_positions = self.model.system.positions




