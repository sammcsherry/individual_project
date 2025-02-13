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
import plotting as plot



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
        self.current_in_history = []
        self.current_out_history = []

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm != 0:
            self.state /= norm

    def evolve(self):
        print(type(self.current_in_history))
        for step in range(self.num_steps):
            self.get_current_step()
            self.state = self.apply_evolution_operator(self.state)
            if self.create_GIF:
                self.save_frames(step)
        self.plot_current_in_out()
        transmission = self.integrate_current_density()
        print(transmission)


    def get_current_step(self):
            crossing_hopping_out = self.get_crossing_hoppings(cross_section_position = 1)
            crossing_hopping_in = self.get_crossing_hoppings(cross_section_position = -5)
            current_in = self.calculate_current(crossing_hopping_in)
            current_out = self.calculate_current(crossing_hopping_out)
            self.current_in_history.append(current_in)
            self.current_out_history.append(current_out)
    
    def integrate_current_density(self):
        return np.sum(self.current_out_history)/np.sum(self.current_in_history)
    
    def plot_current_in_out(self):
        plt.plot(self.current_in_history, label="current into scatter region", color="blue")
        plt.plot(self.current_out_history, label="current out of scatter region", color="red")

        plt.xlabel("time")
        plt.ylabel("current desnity")
        plt.legend()
        plt.title("current over time")
        plt.grid(True)
        plt.show()

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

    def get_crossing_hoppings(self, cross_section_position):

        cross_section = self.lattice_positions[0] > cross_section_position

        hamiltonian = self.lattice.hamiltonian_coo
        
        # Find hoppings where i is on one side and j is on the other
        crossing_mask = cross_section[hamiltonian.row] & ~cross_section[hamiltonian.col]
        crossing_hoppings = {
            'rows': hamiltonian.row[crossing_mask],
            'cols': hamiltonian.col[crossing_mask],
            'data': hamiltonian.data[crossing_mask]
        }

        return crossing_hoppings
    
    def calculate_current(self, crossing_hoppings):
        rows = crossing_hoppings['rows']
        cols = crossing_hoppings['cols']
        data = crossing_hoppings['data']

        psi_i = self.state[rows]
        psi_j = self.state[cols]

        current_density = (1j * data) * (psi_i.conj() * psi_j - psi_j.conj() * psi_i)
        current_density = np.minimum(0, np.real(current_density))  

        return np.sum(current_density)



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

    def get_wave_energy(self, kx, ky):
        hbar_vf = 0.66
        k_magnitude = np.sqrt(kx**2+ky**2)
        self.energy = hbar_vf*k_magnitude

    def plot_wavefunction(self):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]

        wavefunction_values = np.abs(self.state)**2

        plot.Plot3D(x_data = x,
                    y_data= y,
                    z_data= wavefunction_values,
                    x_label= "X Position",
                    y_label= "Y Position",
                    z_label= "",
                    title="Wavefunction Amplitude on Graphene Lattice", type_of_plot="scatter")
    
    


            