from lattice_class import Lattice
from evolve_wavefunction import Wavefunction
import numpy as np

ev = 1

lattice_instance = Lattice(num_unit_cells=1000.5, unit_cell_length=0.24595)

lattice_instance.add_sublattice('A', [0, 0])
lattice_instance.add_hopping([1, 0], 'A', 'A', 2.8*ev)

lattice_instance.create_model()

hamiltonian = lattice_instance.get_hamiltonian()
initial_wavefunction = np.zeros(1000)
initial_wavefunction[0] = 1 
initial_wavefunction = initial_wavefunction.reshape(-1,1)
wavefunction_instance = Wavefunction(initial_wavefunction, hamiltonian, time_step = 0.0001)
wavefunction_instance.evolve(num_steps=1)
evolved_state = wavefunction_instance.get_state()

