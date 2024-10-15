from lattice_class import Lattice
import numpy as np
from evolve_wavefunction import 

lattice_instance = Lattice(num_unit_cells=6, unit_cell_length=0.24595)

# Add sublattices and hoppings
lattice_instance.add_sublattice('A', [0, 0])
lattice_instance.add_hopping([1, 0], 'A', 'A', 2.8)

lattice_instance.create_model()
print(lattice_instance.get_hamiltonian())

initial_wavefunction = np.array([0,0,1,0,0])  # Define your initial wavefunction
hamiltonian = lattice_instance.get_hamiltonian()
wavefunction_instance = Wavefunction(initial_wavefunction, hamiltonian)
# wavefunction_instance.evolve(num_steps=100)
# evolved_state = wavefunction_instance.get_state()

