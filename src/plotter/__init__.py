import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        plt.figure()
        plt.xlabel('Height')
        plt.ylabel('Weight')

    @staticmethod
    def plot_line(x, y):
        plt.plot(x, y)

    @staticmethod
    def plot_points(x, y):
        plt.grid()
        plt.scatter(x, y)

    @staticmethod
    def show():
        plt.show()
