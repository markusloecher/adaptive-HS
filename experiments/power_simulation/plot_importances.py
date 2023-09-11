from matplotlib import pyplot as plt
import os
import joblib
import argparse
import numpy as np


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_importances(result, relevance):
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    fig, ax = plt.subplots()
    importances = result[relevance]
    width = 0.1
    for i, key in enumerate(importances.keys()):
        bpl = ax.boxplot(importances[key], positions=np.arange(5) + (i-2)*width,
                         sym='', widths=width, showfliers=False)
        set_box_color(bpl, colors[i])
        ax.plot([], c=colors[i], label=key)

    ax.legend()
    ax.set_title(f"Relevance: {relevance}")
    ax.set_xticks(np.arange(5), ["X1", "X2", "X3", "X4", "X5"])
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="output/importances.pkl")
    parser.add_argument("--output-dir", type=str, default="plot")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    result = joblib.load(args.input_file)
    for relevance in result.keys():
        fig, ax = plot_importances(result, relevance)

        fig.savefig(os.path.join(
            args.output_dir, f"importances_{relevance}.png"))