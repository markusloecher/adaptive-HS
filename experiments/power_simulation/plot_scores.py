import os
import joblib
import argparse
from matplotlib import pyplot as plt
import numpy as np


def plot_scores(result, relevance):
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    fig, ax = plt.subplots()
    scores = result[relevance]
    print(scores.keys())
    for i, key in enumerate(scores.keys()):
        # Make line plot averaging over rows
        ax.plot(np.mean(scores[key], axis=0), label=key, c=colors[i])
        # Plot confidence interval
        ci = 1.96 * np.std(scores[key], axis=0) / np.sqrt(scores[key].shape[0])
        ax.fill_between(np.arange(len(scores[key][0])),
                        np.mean(scores[key], axis=0) - ci,
                        np.mean(scores[key], axis=0) + ci,
                        alpha=0.2, color=colors[i])
    
    ax.legend()
    ax.set_title(f"Relevance: {relevance}")
    ax.set_xticks(np.arange(7), [0., 0.1, 1.0, 10.0, 25.0, 50.0, 100.0])
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("Accuracy")
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="output/scores.pkl")
    parser.add_argument("--output-dir", type=str, default="plot")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    result = joblib.load(args.input_file)
    for relevance in result.keys():
        fig, ax = plot_scores(result, relevance)

        fig.savefig(os.path.join(
            args.output_dir, f"scores_{relevance}.png"))