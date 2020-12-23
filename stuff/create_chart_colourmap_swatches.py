import numpy as np
import matplotlib.pyplot as plt

# https://matplotlib.org/tutorials/colors/colormaps.html
cmaps = [
    "BrBG",
    "cividis",
    "hot",
    "inferno",
    "magma",
    "PiYG",
    "plasma",
    "PRGn",
    "PuOr",
    "RdBu",
    "RdGy",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "viridis",
    "YlGnBu",
    "YlOrBr"
]

gradient = np.linspace(0, 1, 15)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap):
    fig = plt.figure(figsize=(1.5, 0.15))
    axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap))
    axes.set_axis_off()


for cmap in cmaps:
    plot_color_gradients(cmap)
    plt.savefig("{}.png".format(cmap))
    plt.close()
