import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    for filename in os.listdir(args.results_dir):
        if filename.endswith(".csv"):
            ds_name = filename.split(".")[0]
            df = pd.read_csv(os.path.join(args.results_dir, filename))
            score_key = df.columns[-1]
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x="lambda", y=score_key, hue="shrink_mode")
            plt.xscale("log")
            plt.savefig(
                os.path.join(args.out_dir, f"{ds_name}.svg"),
                bbox_inches="tight",
            )
