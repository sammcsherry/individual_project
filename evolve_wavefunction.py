import numpy as np
from scipy.linalg import expm  # Import for matrix exponential

class Wavefunction:
    def __init__(self, initial_state, hamiltonian, time_step=0.01):
        self.state = np.array(initial_state).reshape(-1, 1)  # Ensure state is a column vector
        self.hamiltonian = hamiltonian  
        self.time_step = time_step

    def evolve(self, num_steps):
        for _ in range(num_steps):
            self.state = self.apply_evolution_operator(self.state)
            self.normalize()  # Normalize the state after each evolution step
            print(self.get_state())


    def apply_evolution_operator(self, state):
        # Compute the evolution operator as U = exp(-iHt/ℏ)
        hbar = 1  # Set ℏ = 1 for simplicity
        U = expm(-1j * self.hamiltonian * self.time_step / hbar)  # Use expm for matrix exponential
        return U @ state  # Evolve the state

    def normalize(self):
        """Normalize the wavefunction."""
        norm = np.linalg.norm(self.state)
        if norm != 0:
            self.state /= norm  # Normalize the state

    def get_state(self):
        return self.state
