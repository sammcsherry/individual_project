import numpy as np
from scipy.linalg import expm
from plotting import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import imageio
from scipy.sparse.linalg import expm_multiply
from scipy import sparse
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.sparse import coo_matrix
import plotting as plots

hbar = np.float32(6.582e-7)


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
        self.cross_section_history = []
        self.angle_bin=0

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm != 0:
            self.state /= norm

    def evolve(self):
        self.get_all_crossing_hoppings()
        for step in range(self.num_steps):
            print(step)
            print(np.sum(np.abs(self.state)**2))
            self.get_current_step()
            self.state = self.apply_evolution_operator(self.state)

            print(np.sum(self.current_in_history) + np.sum(self.current_out_history))
            if (1-np.sum(self.current_in_history) + np.sum(self.current_out_history) <0.0005):
                break
            if self.create_GIF and step%10==0:
                self.save_frames(step)
            if step%20:
                if np.sum(np.abs(self.state)**2) < 0.01:
                    break

        #self.plot_current_in_out()
        self.integrate_current_density()


    def get_current_step(self):
        current_left, angle_bin_left = self.calculate_current(self.crossing_hopping_left, direction=-1)
        current_right, angle_bin_right = self.calculate_current(self.crossing_hopping_right, direction=1)
        current_top_left, angle_bin_top_left = self.calculate_current(self.crossing_hopping_top_left, direction=1)
        current_top_right,angle_bin_top_right = self.calculate_current(self.crossing_hopping_top_right, direction=1)
        current_bottom_left, angle_bin_bottom_left = self.calculate_current(self.crossing_hopping_bottom_left, direction=-1)
        current_bottom_right, angle_bin_bottom_right = self.calculate_current(self.crossing_hopping_bottom_right, direction=-1)

        self.angle_bin = self.angle_bin +angle_bin_left+ angle_bin_right +angle_bin_top_left+ angle_bin_top_right+ angle_bin_bottom_left+ angle_bin_bottom_right
        current_reflection = current_left+current_top_left+current_bottom_left
        current_transmission = current_right+current_top_right+current_bottom_right
        
        self.current_in_history.append(current_transmission)
        self.current_out_history.append(current_reflection)


    def get_all_crossing_hoppings(self):
        x_pos = self.lattice_positions[0]
        y_pos = self.lattice_positions[1]
        x_min = min(x_pos)
        x_max = max(x_pos)
        tranmission_point = self.lattice.width/2
        y_min = min(y_pos)
        y_max = max(y_pos)
        pml_width = 30
        self.crossing_hopping_left = self.get_crossing_hoppings(cross_section_position = x_min+pml_width, axis='x', range_start=y_min+pml_width, range_end=y_max-pml_width)
        self.crossing_hopping_right = self.get_crossing_hoppings(cross_section_position = x_max-pml_width, axis='x', range_start=y_min+pml_width, range_end=y_max-pml_width)
        self.crossing_hopping_top_left = self.get_crossing_hoppings(cross_section_position = y_max-pml_width, axis='y', range_start=x_min+pml_width, range_end=x_min+tranmission_point)
        self.crossing_hopping_top_right = self.get_crossing_hoppings(cross_section_position = y_max-pml_width, axis='y', range_start=x_max-(self.lattice.width-tranmission_point), range_end=x_max-pml_width)
        self.crossing_hopping_bottom_left = self.get_crossing_hoppings(cross_section_position = y_min+pml_width, axis='y', range_start=x_min+pml_width, range_end=x_min+tranmission_point)
        self.crossing_hopping_bottom_right = self.get_crossing_hoppings(cross_section_position = y_min+pml_width, axis='y', range_start=x_max-(self.lattice.width-tranmission_point), range_end=x_max-pml_width)


        reflection_probes = [self.crossing_hopping_left, self.crossing_hopping_top_left, self.crossing_hopping_bottom_left]
        transmission_probes = [self.crossing_hopping_right, self.crossing_hopping_top_right, self.crossing_hopping_bottom_right]
        for probe in reflection_probes:
            probe_positions = probe['rows']
            plt.scatter(x_pos[probe_positions], y_pos[probe_positions],color = 'red', s=40)

        for probe in transmission_probes:
            probe_positions = probe['rows']
            plt.scatter(x_pos[probe_positions], y_pos[probe_positions],color = 'black', s=40)

        plt.show()
        

    def integrate_current_density(self):
        self.reflection = np.abs(np.sum(self.current_out_history))
        self.transmission = np.abs(np.sum(self.current_in_history))
        print(f'transmission: {self.transmission}')
        print(f'reflection: {self.reflection}')
        print(f'total: {self.transmission+self.reflection}')
    
    def plot_current_in_out(self):
        plt.plot(self.current_in_history, label="current into scatter region", color="blue")
        plt.plot(self.current_out_history, label="current out of scatter region", color="red")

        plt.xlabel("time")
        plt.ylabel("current desnity")
        plt.legend()
        plt.title("current over time")
        plt.grid(True)
        #plt.show()

    def save_frames(self, step):
        scatter = self.plot_wavefunction()
        frame_filename = f'temp_wavefunction_step_{step}.png'
        plt.savefig(frame_filename)
        plt.close()
        self.frames.append(frame_filename)

    def apply_evolution_operator(self, state):
        
        coeff = -1j * (self.time_step / hbar)

        new_state = expm_multiply(coeff*self.lattice.hamiltonian, self.state)
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

class Wavefunction2D(Wavefunction):
    def __init__(self, lattice, wave_type, time_step, num_steps, **kwargs):
        super().__init__(lattice, wave_type, time_step, num_steps)
        if wave_type == 'gaussian':
            self.state = self.wavefunction_init(**kwargs)
            self.normalize_wavefunction() 
        else:
            raise ValueError("Unsupported wave type")
        
    def gaussian_wavefunction(self, x0, y0, sigma):
        distance_squared = (self.lattice_positions[0] - x0)**2 + (self.lattice_positions[1] - y0)**2
        return np.exp(-distance_squared / (2 * sigma**2))
    
    def plane_wave(self, kx, ky):
        return np.exp(1j*(kx*self.lattice_positions[0] + ky*self.lattice_positions[1]))
    
    def wavefunction_init(self, x0, y0, sigma, kx, ky):
        gaussian = self.gaussian_wavefunction(x0, y0, sigma)
        plane_wave = self.plane_wave(kx, ky)
        return plane_wave * gaussian
    
    def normalize_wavefunction(self):
        norm = np.sqrt(np.sum(np.abs(self.state)**2)) 
        self.state /= norm 

    def get_wave_energy(self, kx, ky):
        hbar_vf = 0.66
        k_magnitude = np.sqrt(kx**2+ky**2)
        self.energy = hbar_vf*k_magnitude

    def plot_wavefunction(self):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]

        wavefunction_values = np.abs(self.state)**2

        plots.Plot3D(x_data = x,
                    y_data= y,
                    z_data= wavefunction_values,
                    x_label= "X Position",
                    y_label= "Y Position",
                    z_label= "",
                    title="Wavefunction Amplitude on Graphene Lattice", type_of_plot="scatter")
        
    def calculate_current(self, crossing_hoppings, direction):
        rows = crossing_hoppings['rows']
        cols = crossing_hoppings['cols']
        data = crossing_hoppings['data']

        psi_i = self.state[rows]
        psi_j = self.state[cols]

        current_density = ((1j * data)/hbar) * (psi_i.conj() * psi_j - psi_j.conj() * psi_i)*self.time_step
        
        current_density = np.minimum(0, direction*np.real(current_density))
        current_density = np.abs(current_density)

        x_coords = self.lattice_positions[0]
        y_coords = self.lattice_positions[1]
        dx = x_coords[cols]  
        dy = y_coords[cols]
        angles_rad = np.arctan2(dy, dx)
        angles_deg = np.degrees(angles_rad)  
        angles_int = np.floor(angles_deg).astype(int)

        angle_bins = np.zeros(360)
        bin_offset = 180
        np.add.at(angle_bins, (angles_int + bin_offset)%360, np.abs(current_density))

        return np.sum(current_density), angle_bins
    
   
    def get_crossing_hoppings(self, cross_section_position, axis, range_start=None, range_end=None):

        if axis == 'x':
            cross_section = self.lattice_positions[0] > cross_section_position
            perpendicular_positions = self.lattice_positions[1]
        elif axis == 'y':
            cross_section = self.lattice_positions[1] > cross_section_position
            perpendicular_positions = self.lattice_positions[0]
        else:
            raise ValueError("Invalid axis. Must be 'x' or 'y'.")
        if range_start is None:
            range_start = np.min(perpendicular_positions)
        if range_end is None:
            range_end = np.max(perpendicular_positions)

        range_mask = (perpendicular_positions >= range_start) & (perpendicular_positions <= range_end)

        hamiltonian = self.lattice.hamiltonian_coo
        
        crossing_mask = (
            cross_section[hamiltonian.row] & ~cross_section[hamiltonian.col] &
            range_mask[hamiltonian.row] & range_mask[hamiltonian.col]
        )

        crossing_hoppings = {
            'rows': hamiltonian.row[crossing_mask],
            'cols': hamiltonian.col[crossing_mask],
            'data': hamiltonian.data[crossing_mask]
        }
        return crossing_hoppings
    
    

        
        


                