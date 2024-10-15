import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt

class Lattice:
    def __init__(self, num_unit_cells, unit_cell_length):
        self.num_unit_cells = num_unit_cells
        self.unit_cell_length = unit_cell_length
        self.sublattices = [] 
        self.hoppings = []
        self.model = None

    def add_sublattice(self, name, position):
        self.sublattices.append((name, position))

    def add_hopping(self, displacement, sublattice1, sublattice2, strength):
        """Add a hopping between sublattices."""
        self.hoppings.append((displacement, sublattice1, sublattice2, strength))

    def create_unit_cell_1d(self):
        """Create a general unit cell based on added sublattices and hoppings."""
        lattice = pb.Lattice(a1=[self.unit_cell_length, 0])
        
        # Add each sublattice
        for name, position in self.sublattices:
            lattice.add_sublattices((name, position))
        
        # Add each hopping
        for displacement, sub1, sub2, strength in self.hoppings:
            lattice.add_hoppings((displacement, sub1, sub2, strength))
        
        return lattice

    def create_model(self):
        """Create the model based on the current sublattices and hoppings."""
        chain_length = self.num_unit_cells * self.unit_cell_length
        shape = pb.line([0, 0], [chain_length, 0])
        lattice = self.create_unit_cell_1d()
        self.model = pb.Model(lattice, shape)
        return self.model

    def plot_model(self):
        self.model.plot()
        plt.title("Lattice Model")
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.axis('equal')
        plt.show()

    def get_hamiltonian(self):
        return self.model.hamiltonian.todense()

