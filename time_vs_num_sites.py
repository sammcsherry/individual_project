import time
import matplotlib.pyplot as plt
from lattice_class import *
from evolve_wavefunction import *
import numpy as np
from pre_set_structures import *

def measure_time_vs_chain_size(begin_length, end_length, step_size, num_sim_steps):
    chain_lengths = range(begin_length, end_length + 1, step_size)
    times = []

    for length in chain_lengths:
        lattice1D = chain_lattice(hopping_strength=2.8, chain_length=length)
        lattice1D.create_model()
        
        wavefunction1D = Wavefunction1D(lattice=lattice1D, wave_type='gaussian', 
                                         time_step=0.1, num_steps=num_sim_steps, x0=50, sigma=2)
        wavefunction1D.add_complex_absorbing_potential(potential_strength=0, n=0)
        start_time = time.time()
        wavefunction1D.evolve()
        end_time = time.time()
        
        times.append(end_time - start_time)

        print(f"Chain Length: {length}, Time Taken: {times[-1]:.4f} seconds")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(chain_lengths, times, marker='o')
    plt.title('Time Taken vs Chain Size')
    plt.xlabel('Chain Size')
    plt.ylabel('Time Taken (seconds)')
    plt.grid()
    plt.xticks(chain_lengths)
    plt.show()

# Example usage
measure_time_vs_chain_size(begin_length=100, end_length=2500, step_size=100, num_sim_steps=10)
