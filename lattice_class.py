import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix



class Lattice:
    def __init__(self, unit_cell_length):
        self.sites = 0
        self.unit_cell_length = unit_cell_length
        self.hamiltonian = None
        self.sublattices = [] 
        self.hoppings = []
        self.model = None

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
        self.hamiltonian = self.model.hamiltonian
        return self.model
    
    def chain_sites(self):
        chain_length = self.num_sites * self.unit_cell_length
        shape = pb.line([0, 0], [chain_length, 0])  # Single line for 1D
        return shape


class Lattice2D(Lattice):
    def __init__(self, unit_vectors, unit_cell_length=0.24595):
        super().__init__(unit_cell_length)
        self.unit_vectors = unit_vectors


    def create_unit_cell(self):
        a1, a2 = self.unit_vectors[0], self.unit_vectors[1]
        lattice = pb.Lattice(a1=a1, a2=a2)  # 2D specific lattice creation
        return self.configure_lattice(lattice)
    
    def create_rectangle(self, width, height):
        shape_obj = pb.rectangle(x=self.unit_cell_length * width,
                                     y=self.unit_cell_length * height)
        
        lattice = self.create_unit_cell()
        self.model = pb.Model(lattice, shape_obj)
        self.hamiltonian = self.model.hamiltonian
        return self.model
    
        