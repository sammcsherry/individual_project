import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import imageio
from scipy.sparse.linalg import expm_multiply
from scipy import sparse
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.sparse import coo_matrix
import finufft



class Wavefunction:
    def __init__(self, lattice, wave_type, time_step,  num_steps, create_GIF = True):
        self.create_GIF = create_GIF
        self.lattice = lattice
        self.x_length = 0
        self.y_length = 0
        self.wave_type = wave_type
        self.time_step = time_step
        self.num_steps = num_steps
        self.frames = []  # store frames for gif
        self.lattice_positions = lattice.lattice_positions
        #self.num_sites = len(self.lattice_positions)
        self.state = None

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm != 0:
            self.state /= norm

    def evolve(self):
        for step in range(self.num_steps):
            self.state = self.apply_evolution_operator(self.state)
            if self.create_GIF:
                self.save_frames(step)

    def save_frames(self, step):
        scatter = self.plot_wavefunction()
        frame_filename = f'temp_wavefunction_step_{step}.png'
        plt.savefig(frame_filename)
        plt.close()
        self.frames.append(frame_filename)

    def apply_evolution_operator(self, state):
        hbar = 1
        new_state = expm_multiply(-1j * self.lattice.hamiltonian * self.time_step / hbar, self.state)
        return new_state

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
            self.state = self.wavefunction_init(**kwargs)
        
        else:
            raise ValueError("Unsupported wave type")

    def gaussian_wavefunction(self, x0, sigma):
        return np.exp(-((self.lattice_positions[0] - x0) ** 2) / (2 * sigma ** 2))
    
    def plane_wave(self, kx):
        return np.exp(1j*(kx*self.lattice_positions[0]))
    
    def wavefunction_init(self, x0, sigma, kx):
        gaussian = self.gaussian_wavefunction(x0, sigma)
        plane_wave = self.plane_wave(kx)
        return plane_wave*gaussian

    def plot_wavefunction(self):
        plt.figure(figsize=(8, 6))
        
        #plot probability density
        plt.plot(np.abs(self.state)**2)
        plt.xlabel('Position')
        plt.ylabel('Probability Density')
        plt.title(f'1D Wavefunction Evolution')
        plt.legend()

    def calculate_transmission(self):
        probability_density = np.abs(self.k_state)**2 
        N = len(self.state)
        dx = 1  
        k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
        transmitted = np.sum(probability_density[k > 0])
        reflected = np.sum(probability_density[k < 0])
        return transmitted, reflected
    
    def get_momentum_x(self):
        phase = np.angle(self.state)
        kx = np.gradient(phase)
        print(self.state)
        print(kx)

    def calculate_transmission_reflection(self):
        # Extract lattice positions
        positions = self.lattice_positions[0]  # 1D lattice positions

        # Scale positions for NUFFT
        positions_scaled = 2 * np.pi * (positions - positions.min()) / (positions.max() - positions.min())

        # Perform NUFFT
        n_modes = len(self.state) 
        momentum_components = finufft.nufft1d1(positions_scaled, self.state, n_modes)

        # Define symmetric k-values
        real_space_extent = positions.max() - positions.min()  # Total extent in real space
        delta_k = 2 * np.pi / real_space_extent  # Momentum space resolution
        k_values = np.linspace(-n_modes / 2 * delta_k, (n_modes / 2 - 1) * delta_k, n_modes)

        # Remove repeated peaks (limit k-range to Nyquist cutoff)
        k_cutoff = np.pi / np.min(np.diff(np.sort(positions))) 
        valid_indices = np.abs(k_values) <= k_cutoff
        filtered_k_values = k_values[valid_indices]
        filtered_momentum = momentum_components[valid_indices]
        filtered_velocity = np.sin(filtered_k_values)

        probabilities = np.abs(filtered_momentum)**2

        transmitted = np.sum(probabilities[filtered_k_values > 0])

        reflected = np.sum(probabilities[filtered_k_values < 0])

        plt.plot(filtered_velocity, probabilities)
        plt.title("wave function in k space")
        plt.xlabel("momentum")
        plt.ylabel("probability")
        plt.show()

        return transmitted, reflected





class Wavefunction2D(Wavefunction):
    def __init__(self, lattice, wave_type, time_step, num_steps, **kwargs):
        super().__init__(lattice, wave_type, time_step, num_steps)
        if wave_type == 'gaussian':
            self.state = self.wavefunction_init(**kwargs)
        else:
            raise ValueError("Unsupported wave type")
        #print(self.lattice.get_hamiltonian().shape)
        
    def gaussian_wavefunction(self, x0, y0, sigma):
        distance_squared = (self.lattice_positions[0] - x0)**2 + (self.lattice_positions[1] - y0)**2
        return np.exp(-distance_squared / (2 * sigma**2))
    
    def plane_wave(self, kx, ky):
        return np.exp(1j*(kx*self.lattice_positions[0] + ky*self.lattice_positions[1]))
    
    def wavefunction_init(self, x0, y0, sigma, kx, ky):
        gaussian = self.gaussian_wavefunction(x0, y0, sigma)
        plane_wave = self.plane_wave(kx, ky)
        return plane_wave*gaussian



    def plot_wavefunction(self):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]

        wavefunction_values = np.abs(self.state)**2

        # Create scatter plot
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(x, y, c=wavefunction_values, cmap="viridis", s=40)  # `s` adjusts point size

        # Add colorbar to represent amplitude values
        plt.colorbar(scatter, label="Wavefunction Amplitude")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Wavefunction Amplitude on Graphene Lattice")
        plt.axis("equal")  # Ensures equal scaling for x and y axes

