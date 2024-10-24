import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import imageio
from scipy.sparse.linalg import expm_multiply

class Wavefunction:
    def __init__(self, lattice, wave_type, time_step,  num_steps):
        self.lattice = lattice
        self.wave_type = wave_type
        self.time_step = time_step
        self.num_steps = num_steps
        self.frames = []  # store frames for gif

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm != 0:
            self.state /= norm

    def evolve(self):
        for step in range(self.num_steps):
            self.state = self.apply_evolution_operator(self.state)
            
            self.normalize()
            
            frame_filename = self.plot_wavefunction(step)
            self.frames.append(frame_filename)

    def apply_evolution_operator(self, state):
        hbar = 1
        U = expm_multiply(-1j * self.lattice.hamiltonian * self.time_step / hbar, state)
        return U



    def create_gif(self):
        with imageio.get_writer('wavefunction_evolution.gif', mode='I', duration=0.1) as writer:
            for frame in self.frames:
                image = imageio.imread(frame)
                writer.append_data(image)
        print("GIF saved as wavefunction_evolution.gif")

class Wavefunction1D(Wavefunction):
    def __init__(self, lattice, wave_type, time_step, num_steps, **kwargs):
        super().__init__(lattice, wave_type, time_step, num_steps)
        if wave_type == 'gaussian':
            self.state = self.gaussian_wavefunction(**kwargs)
        elif wave_type == 'delta':
            self.state = self.delta_wavefunction(**kwargs)
        else:
            raise ValueError("Unsupported wave type")

    def gaussian_wavefunction(self, x0, sigma):
        num_sites = self.lattice.get_num_sites()
        x = np.arange(num_sites)
        
        # 1D Gaussian
        wavefunction_values = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        
        return wavefunction_values

    def plot_wavefunction(self, step):
        plt.figure(figsize=(8, 6))
        
        #plot probability density
        plt.plot(np.abs(self.state)**2, label=f'Step {step}')
        plt.xlabel('Position')
        plt.ylabel('Probability Density')
        plt.title(f'1D Wavefunction Evolution - Step {step}')
        plt.legend()
        
        frame_filename = f'temp_wavefunction_1D_step_{step}.png'
        plt.savefig(frame_filename)
        plt.close()
        
        return frame_filename
    
    def add_complex_absorbing_potential(self, potential_strength, n):
        num_sites = self.lattice.get_num_sites()
        H = self.lattice.get_hamiltonian().astype(np.complex128)
        complex_potential = np.zeros(num_sites, dtype=complex)

        # apply to the first n sites
        for i in range(min(n, num_sites)):
            complex_potential[i] = -1j * potential_strength * (n-i)/n

        # apply to the last n sites
        for i in range(min(n, num_sites)):
            complex_potential[-(i + 1)] = -1j * potential_strength * ((n-i)/n)

        # Update the Hamiltonian
        H += np.diag(complex_potential)
        self.lattice.set_hamiltonian(H)

    
class Wavefunction2D(Wavefunction):
    def __init__(self, lattice, wave_type, time_step, num_steps, **kwargs):
        super().__init__(lattice, wave_type, time_step, num_steps)
        if wave_type == 'gaussian':
            self.state = self.gaussian_wavefunction(**kwargs)
        else:
            raise ValueError("Unsupported wave type")
        self.evolve()

    def gaussian_wavefunction(self, x0, y0, sigma):
        side_length = int(np.sqrt(self.lattice.get_num_sites()))  # assuming a square lattice
        x = np.arange(side_length)
        y = np.arange(side_length)
        X, Y = np.meshgrid(x, y)

        # 2D gaussian wavefunction
        wavefunction_values = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
        
        return wavefunction_values.ravel()
    
    def plot_wavefunction(self, step):
        plt.figure(figsize=(8, 6))
        num_sites = self.lattice.get_num_sites()
        side_length = int(np.sqrt(num_sites))  # assuming square lattice

        # reshape state to 2D only for plotting
        wavefunction_values = np.abs(self.state.reshape(side_length, side_length))**2  # probability density

        # create meshgrid for plotting
        x = np.linspace(0, side_length - 1, side_length)
        y = np.linspace(0, side_length - 1, side_length)
        X, Y = np.meshgrid(x, y)

        # 3D plot
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, wavefunction_values, cmap='viridis', edgecolor='none')
        
        ax.set_title(f'Wavefunction Evolution - Step {step}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Probability Density')
        
        frame_filename = f'temp_wavefunction_step_{step}.png'
        plt.savefig(frame_filename)
        plt.close()
        
        return frame_filename

