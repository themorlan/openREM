import numpy as np
import matplotlib.pyplot as plt

# https://matplotlib.org/tutorials/colors/colormaps.html
cmaps = [
    # Sequential maps
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',

    # Sequential2 maps
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper',

    # Diverging
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',

    # Miscellaneuos
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
    'gist_ncar'
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
