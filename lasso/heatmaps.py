from fileinput import filename
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_epsilon_from_file(filename):
    """
    Reads the epsilon array from a file and returns it as a numpy array.
    """

    epsilon = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        dims = list(map(int, lines[0].strip().split()))
        for i in range(1,len(lines)):
            epsilon.append(float(lines[i].strip()))
        epsilon = np.array(epsilon)
        epsilon = epsilon.reshape(dims[0],dims[1],dims[2])

    return epsilon

def format_epsilon_heatmap(epsilon):
    """
    Reformats the epsilon array to be compatible with the heatmap plotting function.
    """
    formatted_epsilon = np.array(np.zeros((epsilon.shape[1],epsilon.shape[0],epsilon.shape[2]))) #shape = (M,D,L)
    for d in range(epsilon.shape[0]):
        for m in range(epsilon.shape[1]):
            for l in range(epsilon.shape[2]):
                formatted_epsilon[m,d,l] = epsilon[d,m,l]

    return formatted_epsilon

def epsilon_heatmap(formatted_epsilon, name):
    """
    Plots the epsilon array as a heatmap and saves it to a file.
    """

    m, d, l = formatted_epsilon.shape

    fig, axes = plt.subplots(1, m, figsize=(4*m, 4))
    for i in range(m):
        sns.heatmap(
            formatted_epsilon[i],
            ax=axes[i],
            cmap="viridis",
            cbar=(i == m-1), # show colorbar only once
            annot = True,
            fmt = ".2f",
            annot_kws={"fontsize": 6},
            vmin = np.min(formatted_epsilon),
            vmax = np.max(formatted_epsilon)
            )
        axes[i].set_title("M = "+str(i)+" Heatmap")
    plt.tight_layout()
    plt.savefig(name)
    plt.close('all')

    return None


def heatmap_from_array(epsilon, name):
    formatted_epsilon = format_epsilon_heatmap(epsilon)
    epsilon_heatmap(formatted_epsilon, name)

    return None

