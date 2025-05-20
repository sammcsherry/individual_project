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
from scipy.interpolate import griddata
import os



hbar = np.float32(6.582e-16)


class Wavefunction:
    def __init__(self, lattice, wave_type, time_step,  num_steps, charge_type,charge_frequency, create_GIF = True):
        self.charge_frequency = charge_frequency
        self.charge_type = charge_type
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
        self.x = self.lattice_positions[0]
        self.y = self.lattice_positions[1]
        self.state_crop_mask = np.ones(len(self.x), dtype=bool)
        self.step=0

        self.state = None
        self.current_in_history = []
        self.current_out_history = []
        self.cross_section_history = []
        self.amplitude_history = []
        self.amplitude_history_sum = []
        self.angle_bin=0
        self.transfer_cross_section=0
        coeff = -1j * self.time_step / (2*hbar)
        diag_potential = self.lattice.potential_hamiltonian.diagonal()
        self.exp_potential = np.exp(coeff*diag_potential)

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm != 0:
            self.state /= norm

    def evolve(self):
        for step in range(self.num_steps):
            #self.update_coulomb_potential(start_amplitude=0.4, end_amplitude=0.6)
            self.step=step
            self.crop_hamiltonian()
            if self.create_GIF and step%10==0:
                self.save_frames(step)


            self.calculate_standard_deviations()
            self.apply_evolution_operator(self.state)

    def update_coulomb_potential(self, start_amplitude, end_amplitude):
        mid = 0.5*(start_amplitude+end_amplitude)
        range = end_amplitude-start_amplitude
        period = 200/self.charge_frequency
        beta = range * np.sin(2 * np.pi * self.time_step / period) + mid
        self.lattice.add_coulomb_potential(x0=0, y0=0, beta=beta)


    def get_current_step(self):
        crossing_hopping_left, angle_bin_left = self.calculate_current(self.crossing_hopping_left, direction=-1)
        crossing_hopping_right, angle_bin_right = self.calculate_current(self.crossing_hopping_right, direction=1)
        crossing_hopping_lower_right, angle_bin_top_left = self.calculate_current(self.crossing_hopping_lower_right, direction=-1)
        crossing_hopping_upper_right,angle_bin_top_right = self.calculate_current(self.crossing_hopping_upper_right, direction=1)
        crossing_hopping_lower_left, angle_bin_bottom_left = self.calculate_current(self.crossing_hopping_lower_left, direction=-1)
        crossing_hopping_upper_left, angle_bin_bottom_right = self.calculate_current(self.crossing_hopping_upper_left, direction=1)

        self.angle_bin = self.angle_bin +angle_bin_left+ angle_bin_right +angle_bin_top_left+ angle_bin_top_right+ angle_bin_bottom_left+ angle_bin_bottom_right
        current_transmission = crossing_hopping_right+crossing_hopping_lower_right+crossing_hopping_upper_right
        current_reflection = crossing_hopping_left+crossing_hopping_lower_left+crossing_hopping_upper_left
        
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
        pml_width_T = 250
        pml_width_R = 200
        self.crossing_hopping_left = self.get_crossing_hoppings(cross_section_position = x_min+pml_width_R, axis='x', range_start=y_min+pml_width_R, range_end=y_max-pml_width_R)
        self.crossing_hopping_right = self.get_crossing_hoppings(cross_section_position = x_max-pml_width_T, axis='x', range_start=y_min+pml_width_R, range_end=y_max-pml_width_R)
        self.crossing_hopping_upper_left = self.get_crossing_hoppings(cross_section_position = y_max-pml_width_R, axis='y', range_start=x_min+pml_width_R, range_end=x_min+tranmission_point)
        self.crossing_hopping_upper_right = self.get_crossing_hoppings(cross_section_position = y_max-pml_width_R, axis='y', range_start=x_max-(self.lattice.width-tranmission_point), range_end=x_max-pml_width_T)
        self.crossing_hopping_lower_left = self.get_crossing_hoppings(cross_section_position = y_min+pml_width_R, axis='y', range_start=x_min+pml_width_R, range_end=x_min+tranmission_point)
        self.crossing_hopping_lower_right = self.get_crossing_hoppings(cross_section_position = y_min+pml_width_R, axis='y', range_start=x_max-(self.lattice.width-tranmission_point), range_end=x_max-pml_width_T)
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
        os.makedirs("image_dump", exist_ok=True)
        
        frame_filename = f'image_dump/temp_wavefunction_step_{step}.png'
        scatter = self.plot_wavefunction()
        plt.savefig(frame_filename)
        plt.close()
        
        self.frames.append(frame_filename)

    def apply_evolution_operator(self, state):
        cropped_hamiltonian = self.lattice.hamiltonian[np.ix_(self.state_crop_mask, self.state_crop_mask)]
        coeff = -1j * (self.time_step / hbar)
        cropped_exp_potential = self.exp_potential[self.state_crop_mask]
        #new_cropped_state = cropped_exp_potential*(self.state[self.state_crop_mask])
        new_cropped_state = expm_multiply(coeff*cropped_hamiltonian, self.state[self.state_crop_mask])
        #new_cropped_state = cropped_exp_potential*(new_cropped_state)

        self.state[self.state_crop_mask] = new_cropped_state


    def create_gif(self, gif_name):
        with imageio.get_writer(f'{gif_name}.gif', mode='I', duration=0.1) as writer:
            for frame in self.frames:
                image = imageio.imread(frame)
                writer.append_data(image)
        print("GIF saved as wavefunction_evolution.gif")

    def crop_hamiltonian(self):
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]
        self.state_crop_mask = np.abs(self.state)**2 > 1e-7
        self.cropped_x = x[self.state_crop_mask]
        self.cropped_y = y[self.state_crop_mask]
        max_x = max(self.cropped_x) + 50
        min_x = min(self.cropped_x) - 50
        max_y = max(self.cropped_y) + 50
        min_y = min(self.cropped_y) -50
        self.state_crop_mask = (x < max_x ) & (x > min_x) & (y < max_y ) & (y > min_y)
        self.cropped_state = self.state[self.state_crop_mask]
        self.cropped_hamiltonian = self.lattice.hamiltonian[np.ix_(self.state_crop_mask, self.state_crop_mask)]
        print(self.cropped_hamiltonian.shape)
        
        
    



class Wavefunction2D(Wavefunction):
    def __init__(self, lattice, wave_type, time_step, num_steps, charge_type, charge_frequency, **kwargs):
        super().__init__(lattice, wave_type, time_step, num_steps, charge_type, charge_frequency)
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]
        
        if wave_type == 'gaussian':
            self.state = self.wavefunction_init(self.charge_type,x,y,**kwargs)
            self.normalize_wavefunction() 
        elif wave_type == 'bilayer':
            print(self.lattice.model.system.sublattices)
            mask = (self.lattice.model.system.sub == 0) | (self.lattice.model.system.sub ==3)
            self.state = self.wavefunction_init(self.charge_type, x,y,**kwargs)
            self.normalize_wavefunction() 
            #self.state[mask] = 0
        else:
            raise ValueError("Unsupported wave type")
        
    def gaussian_wavefunction(self, x0, y0, sigma, x , y):
        self.sigma = sigma
        distance_squared = (x - x0)**2 + (y - y0)**2
        return np.exp(-distance_squared / (2 * sigma**2))
    
    def plane_wave(self, kx, ky, x, y):
        return np.exp(1j*(kx*x + ky*y))
    
    def wavefunction_init(self, charge_type, x, y, x0, y0, sigma, kx, ky):
        charge=1
        if charge_type == 'electron':
            1
        elif charge_type == 'hole':
            -1
        length = len(x)
        phase_A = np.ones(length, dtype=complex)
        phase_B = np.ones(length, dtype=complex)
        B_lattice_mask = self.lattice.model.system.sublattices == 1
        phase_B[B_lattice_mask] = 1/np.sqrt(2)
        A_lattice_mask = self.lattice.model.system.sublattices == 0
        phase_A[A_lattice_mask] = 1/np.sqrt(2)

        gaussian = self.gaussian_wavefunction(x0, y0, sigma, x, y)
        plane_wave = self.plane_wave(kx, ky, x, y)
        return plane_wave * gaussian *  phase_B * phase_A
    
    def chiral_phase_factor(self, kx, ky):
        theta = np.arctan2(ky, kx)
        return np.exp(1j * theta)
    
    def normalize_wavefunction(self):
        norm = np.sqrt(np.sum(np.abs(self.state)**2)) 
        self.state /= norm 

    def get_wave_energy(self, kx, ky):
        hbar_vf = 0.66
        k_magnitude = np.sqrt(kx**2+ky**2)
        self.energy = hbar_vf*k_magnitude

    def crop_lattice(self, r):
        
        x = self.lattice_positions[0]
        y = self.lattice_positions[1]
        wavefunction_values = np.abs(self.state)**2

        x = x[self.state_crop_mask]
        y = y[self.state_crop_mask]
        wavefunction_values = wavefunction_values[self.state_crop_mask]


        mask = np.sqrt(x**2+y**2)<r
        x = x[mask]
        y = y[mask]
        wavefunction_values = wavefunction_values[mask]

        return x, y, wavefunction_values
    
    def get_amplitude_history(self):
        x,y,amplitude = self.crop_lattice(r=0.2)
        self.amplitude_history.append(amplitude)
        self.amplitude_history_sum.append(np.sum(amplitude))
        with open('coulomb_amplitudes_0.txt', 'a') as file:
            file.write(f"{np.sum(amplitude):.10f}\t")
            file.write('\n')

        
    def plot_decay_rate(self):
        plt.plot(self.amplitude_history_sum)
        plt.show()

    def get_energy_spectrum(self):
        self.amplitude_history = np.array(self.amplitude_history)
        n_rows = len(self.amplitude_history)
        fft_cumsum_results = np.zeros(n_rows, dtype=np.complex128)
        for column in self.amplitude_history.T:
            fft_column = np.fft.fft(column)
            fft_cumsum_results += fft_column
        dt = 0.1
        freq = np.fft.fftfreq(n_rows, d=dt)
        plt.stem(freq, np.abs(fft_cumsum_results))
        plt.xlabel("Frequency (Hz)" if dt != 1.0 else "Frequency (1/units)")
        plt.ylabel("Magnitude")
        plt.title("Cumulative FFT Sum")
        plt.grid(True)
        plt.show()

    def plot_wavefunction(self):
        x = self.x[self.state_crop_mask]
        y = self.y[self.state_crop_mask]
        amplitudes = np.abs(self.state[self.state_crop_mask])**2
        #x, y, wavefunction_values = self.crop_lattice(r=0.2)

        plots.Plot3D(x_data = x,
                    y_data= y,
                    z_data= amplitudes,
                    x_label= "$X$",
                    y_label= "$Y$",
                    z_label= "",
                    title="Wavefunction Amplitude on Graphene Lattice", type_of_plot="scatter")
        
    def plot_wavefunction_bilayer(self):
        top_layer_mask = (self.lattice.model.system.sublattices == 0) | (self.lattice.model.system.sublattices == 3)
        top_layer_x = self.lattice_positions[0][top_layer_mask]
        top_layer_y = self.lattice_positions[1][top_layer_mask]

        bottom_layer_mask = (self.lattice.model.system.sub == 1) | (self.lattice.model.system.sub == 2)
        bottom_layer_x = self.lattice_positions[0][bottom_layer_mask]
        bottom_layer_y = self.lattice_positions[1][bottom_layer_mask]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) 

        ax1.scatter(top_layer_x, top_layer_y, c=np.abs(self.state[top_layer_mask])**2, cmap="viridis", s=40)

        ax2.scatter(bottom_layer_x, bottom_layer_y, c=np.abs(self.state[bottom_layer_mask])**2, cmap="viridis", s=40)

        plt.tight_layout()

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
        angles_rad = np.arctan2(dy,dx)
        self.transfer_cross_section += np.sum((1-np.cos(angles_rad))*current_density)
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
    
    
    def interpolate(self, width, height, charge):
        import numpy.fft as fft
        k_center = (17.28, 0)
        zoom_range = 0.6

        x = self.lattice_positions[0]
        y = self.lattice_positions[1]
        amplitude = self.state 
        
       
        grid_resolution = 0.1  
        
        x_min, x_max = -width/2, width/2
        y_min, y_max = -height/2, height/2

        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        
        grid_points = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([grid_points[0].ravel(), grid_points[1].ravel()]).T
        positions = np.vstack([x, y]).T

        amplitude_grid = griddata(positions, amplitude, grid_points, method='linear')
        amplitude_grid = amplitude_grid.reshape(len(y_grid), len(x_grid))
        amplitude_grid = np.nan_to_num(amplitude_grid)  

        pad_factor = 4 
        padded_size = (amplitude_grid.shape[0] * pad_factor, amplitude_grid.shape[1] * pad_factor)
        amplitude_padded = np.zeros(padded_size, dtype=complex)
        
        y_start = (padded_size[0] - amplitude_grid.shape[0]) // 2
        x_start = (padded_size[1] - amplitude_grid.shape[1]) // 2
        amplitude_padded[y_start:y_start+amplitude_grid.shape[0], 
                        x_start:x_start+amplitude_grid.shape[1]] = amplitude_grid

        amplitude_k = fft.fftshift(fft.fft2(amplitude_padded))
        
        kx = fft.fftshift(fft.fftfreq(padded_size[1], d=grid_resolution)) * 2 * np.pi
        ky = fft.fftshift(fft.fftfreq(padded_size[0], d=grid_resolution)) * 2 * np.pi
        kx_grid, ky_grid = np.meshgrid(kx, ky)

        norm = np.sqrt(np.sum(np.abs(amplitude_k)**2))
        amplitude_k /= norm
        squared_amplitude_k = np.abs(amplitude_k)**2
        print(f'Total prob k: {np.sum(squared_amplitude_k)}')

        kx_min, kx_max = k_center[0] - zoom_range, k_center[0] + zoom_range
        ky_min, ky_max = k_center[1] - zoom_range, k_center[1] + zoom_range
        
        mask = (kx_grid >= kx_min) & (kx_grid <= kx_max) & \
            (ky_grid >= ky_min) & (ky_grid <= ky_max)
        
        zoom_kx = kx_grid[mask]
        zoom_ky = ky_grid[mask]
        zoom_amp = squared_amplitude_k[mask]
        
        zoom_kx_unique = np.unique(zoom_kx)
        zoom_ky_unique = np.unique(zoom_ky)
        
        zoom_amp_grid = zoom_amp.reshape(len(zoom_ky_unique), len(zoom_kx_unique))
        
        plt.figure(figsize=(8, 6))
        plot = plt.pcolormesh(zoom_kx_unique, zoom_ky_unique, zoom_amp_grid, 
                            shading='auto', cmap='viridis')
        
        plt.colorbar(plot, label='Intensity')
        plt.xlabel('$k_x$ (1/Å)')
        plt.ylabel('$k_y$ (1/Å)')
        plt.title(f'k-space region around ({k_center[0]:.2f}, {k_center[1]:.2f}) ± {zoom_range:.2f}')
        plt.legend()
        import time
        current_time = time.time()  
        plt.savefig(f'k-space step {self.step} {charge}.png')
        
        return zoom_kx, zoom_ky, zoom_amp_grid
        

    def get_transfer_cross_section(self, width, height, charge):
        kx_grid, ky_grid, amplitude_grid = self.interpolate(width=width, height=height, charge=charge)
        

        kx_grid = kx_grid - 17.03
        print(np.mean(kx_grid))
        
        angles_grid = np.arctan2(ky_grid, kx_grid)
        
        angles = angles_grid.ravel() 
        angles = angles-0.52918383
        amplitude = amplitude_grid.ravel()

        self.transfer_cross = np.sum((1-np.cos(angles))*amplitude)


        with open('transfer_cross.txt', 'a') as file:
            file.write(f"{self.transfer_cross:.6f}\t\n")

            file.write('\n')

        angle_bins = np.linspace(-np.pi, np.pi, 360) 
        angle_centers = 0.5 * (angle_bins[1:] + angle_bins[:-1])
        
        bin_indices = np.digitize(angles, angle_bins) - 1
        summed_amplitude = np.zeros_like(angle_centers)
        
        for i in range(len(angle_centers)):
            mask = (bin_indices == i)
            summed_amplitude[i] = np.sum(amplitude[mask])
        dtheta = angle_bins[1] - angle_bins[0] 

        plt.figure(figsize=(10, 6))
        plt.bar(angle_centers, summed_amplitude,width=1, alpha=0.7, edgecolor='k')
        plt.xlabel("Scattering Angle θ (radians)")
        plt.ylabel("Probability Density $|ψ(θ)|^2$")
        plt.title("Angular Distribution of Scattered Electrons")
        plt.grid(True)
        plt.savefig(f'scatter_distrubution {charge}.png')
        
        return angle_centers, summed_amplitude
        
    def calculate_standard_deviations(self):
            
            x_pos = self.lattice_positions[0][self.state_crop_mask]
            y_pos = self.lattice_positions[1][self.state_crop_mask]
            amplitudes_old = self.state[self.state_crop_mask]


    
            crop_mask = np.abs(amplitudes_old)**2 > 1e-9
            x = x_pos[crop_mask]
            y = y_pos[crop_mask]
            amplitudes = amplitudes_old[crop_mask]
            density = np.abs(amplitudes)**2

            self.mean_x = np.sum(x * density)
            self.mean_y = np.sum(y * density)
            
            mean_x2 = np.sum(x**2 * density)
            mean_y2 = np.sum(y**2 * density)
            
            self.standard_div_x = np.sqrt(mean_x2 - self.mean_x**2)
            self.standard_div_y = np.sqrt(mean_y2 - self.mean_y**2)

            with open(f'Gaussian-Track{self.sigma}.txt', 'a') as file:
                file.write(f"{self.mean_x:.2f}\t{self.mean_y:.6f}\t{self.standard_div_x:.6f}\t{self.standard_div_y:.6f}\t\n")



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
