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
        self.lattice_positions = lattice.model.system.positions
        self.num_sites = len(self.lattice_positions)
        self.k_state = None

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
        frame_filename = self.plot_wavefunction(step)
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
            self.state = self.gaussian_wavefunction(**kwargs)
        elif wave_type == 'delta':
            self.state = self.delta_wavefunction(**kwargs)
        else:
            raise ValueError("Unsupported wave type")

    def gaussian_wavefunction(self, x0, sigma):
        num_sites = self.lattice.get_num_sites()
        x = np.arange(num_sites)
        wavefunction_values = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        # Normalize the wavefunction
        wavefunction_values /= np.sqrt(np.sum(np.abs(wavefunction_values)**2))
        return wavefunction_values
    
    def spinor_state_init(self, x_momentum, y_momentum):
        theta = np.arctan(y_momentum/x_momentum)
        spinor_state = np.array([np.exp(1j*theta/2), np.exp(-1j*theta/2)])
        return spinor_state.reshape(2,1)
    
    def momentum_state_init(self, k0, sigma, x_momentum, y_momentum):
        spinor_state = self.spinor_state_init(x_momentum, y_momentum)
        gaussian = self.gaussian_wavefunction(k0, sigma).reshape(1,-1)
        self.k_state = spinor_state*gaussian

    def momentum_to_real_space(self):
        # Perform IFFT for each sublattice
        A_real_space = np.fft.ifft(self.k_state)
        B_real_space = np.fft.ifft(self.k_state)

        print(A_real_space)
        print(B_real_space)
        # Split lattice into A- and B-sublattice sites
        #real_space_A = psi_A_real_space[::2]  # Every other site for A-sublattice
        #real_space_B = psi_B_real_space[1::2]  # Remaining sites for B-sublattice



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
        
        return frame_filenamebut 

    def momentum_space(self):
        self.k_state = np.fft.fft(self.state)
        self.k_state = np.fft.fftshift(self.k_state) 

        N = len(self.state)
        # Normalize the FFT output
        self.k_state /= np.sqrt(len(self.state))
        dx = 1 
        k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
        # Plot momentum-space wavefunction
        plt.plot(k, np.abs(self.k_state)**2)
        plt.xlabel('Momentum (k)')
        plt.ylabel('|ψ(k)|^2')
        plt.title('Momentum-Space Wavefunction')
        plt.show()

    def calculate_transmission(self):
        probability_density = np.abs(self.k_state)**2 
        N = len(self.state)
        dx = 1  
        k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
        transmitted = np.sum(probability_density[k > 0])
        reflected = np.sum(probability_density[k < 0])
        print(transmitted)
        print(reflected)
        return transmitted, reflected


class Wavefunction2D(Wavefunction):
    def __init__(self, lattice, wave_type, time_step, num_steps, **kwargs):
        super().__init__(lattice, wave_type, time_step, num_steps)
        if wave_type == 'gaussian':
            self.state = self.gaussian_wavefunction(**kwargs)
        else:
            raise ValueError("Unsupported wave type")
        #print(self.lattice.get_hamiltonian().shape)
        


    def gaussian_wavefunction(self, x0, y0, sigma):
        n = self.lattice.get_num_sites()
        psi_1d = np.zeros(n)

        for i, (x, y) in enumerate(zip(self.lattice_positions[0], self.lattice_positions[1])):
            # Calculate the squared distance from the center
            distance_squared = (x - x0)**2 + (y-y0)**2
            # Assign Gaussian amplitude based on distance
            psi_1d[i] = np.exp(-distance_squared / (2 * sigma**2))

        self.state = psi_1d
        return self.state

    
    def plot_wavefunction(self, step):
        plt.figure(figsize=(8, 6))
        num_sites = self.lattice.get_num_sites()
        side_length = int(np.sqrt(num_sites))  # assuming square lattice

        # reshape state to 2D only for plotting
        wavefunction_values = np.abs(self.state.reshape(self.x_length, self.y_length))**2  # probability density

        # create meshgrid for plotting
        x = np.linspace(0, self.y_length - 1, self.y_length)
        y = np.linspace(0, self.x_length - 1, self.x_length)
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
    
    def plot_wavefunction(self, step):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]

        wavefunction_values = np.abs(self.state)**2

        # Create scatter plot
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(x, y, c=wavefunction_values, cmap="viridis", s=80)  # `s` adjusts point size

        # Add colorbar to represent amplitude values
        plt.colorbar(scatter, label="Wavefunction Amplitude")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Wavefunction Amplitude on Graphene Lattice")
        plt.axis("equal")  # Ensures equal scaling for x and y axes
        frame_filename = f'temp_wavefunction_step_{step}.png'
        plt.savefig(frame_filename)
        plt.close()
        return frame_filename
    
    