import matplotlib.pyplot as plt

class Plotting:
    def __init__(self, x_data, x_label, y_label, title):
        self.x_data = x_data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

class Plot2D(Plotting):
    def __init__(self, x_data, x_label, y_label, title, type_of_plot):
        super().__init__(x_data, x_label, y_label, title)
        if type_of_plot == "scatter":
            self.plot_scatter()
        elif type_of_plot == "line":
            self.plot_line()

    def plot_scatter(self):
        plt.scatter(self.x_data, c='b', marker='o')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)

    def plot_line(self):
        plt.plot(self.x_data)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        print(self.title)
        plt.title(self.title)

class Plot3D(Plotting):
    def __init__(self, x_data, y_data, z_data, x_label, y_label, z_label, title, type_of_plot):
        super().__init__(x_data, x_label, y_label, title)
        self.y_data = y_data
        self.z_data = z_data
        self.z_label = z_label

        if type_of_plot == "scatter":
            self.plot_scatter()

    def plot_scatter(self):
        scatter = plt.scatter(self.x_data, self.y_data, c=self.z_data, cmap="viridis", s=40)
        plt.colorbar(scatter, label="Wavefunction Amplitude")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.axis("equal")

