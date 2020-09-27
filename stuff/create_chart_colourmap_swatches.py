import numpy as np
import matplotlib.pyplot as plt

cmaps = [
    "RdYlBu",
    "RdYlGn",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Spectral"
]

gradient = np.linspace(0, 1, 15)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap):
    fig = plt.figure(figsize=(1, 0.1))
    axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    axes.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap))
    axes.set_axis_off()


for cmap in cmaps:
    plot_color_gradients(cmap)
    plt.savefig("{}.png".format(cmap))
