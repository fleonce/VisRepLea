from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from with_argparse import with_argparse

mpl.rcParams["text.usetex"] = True


@with_argparse
def mse_over_time(
    ijepa_files: list[Path],
    clip_files: list[Path],
    dataset: str,
    out_file: Optional[Path] = None,
):
    """
    Generate the boxplot for MSE improving over time comparing CLIP vs. I-JEPA

    Author: Moritz
    """
    ijepa_files = list(sorted(ijepa_files))

    fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(8, 3.5), sharey=True)
    cmap = mpl.colormaps["tab20c"].colors

    ticks = list(range(len(ijepa_files)))
    labels = [f"{i * 10}K" for i in range(1, len(ticks))] + ["Post Train"]
    ijepa = torch.stack([torch.load(file, weights_only=True) for file in ijepa_files])
    clip = torch.stack([torch.load(file, weights_only=True) for file in clip_files])
    # assert False, ijepa.shape
    ax[0].boxplot(
        ijepa.t(),
        positions=ticks,
        widths=0.85,
        patch_artist=True,
        showmeans=False,
        showfliers=True,
        sym="*",
        flierprops={"color": "black", "linewidth": 0.25, "alpha": 0.25},
        medianprops={"color": "black", "linewidth": 0.5},
        boxprops={"facecolor": cmap[4], "edgecolor": "black", "linewidth": 0.5},
        whiskerprops={"color": "black", "linewidth": 0.5},
        capprops={"color": "black", "linewidth": 0.5},
    )

    ax[1].boxplot(
        clip.t(),
        positions=ticks,
        widths=0.85,
        patch_artist=True,
        showmeans=False,
        showfliers=True,
        sym="*",
        flierprops={"color": "black", "linewidth": 0.25, "alpha": 0.25},
        medianprops={"color": "black", "linewidth": 0.5},
        boxprops={"facecolor": cmap[12], "edgecolor": "black", "linewidth": 0.5},
        whiskerprops={"color": "black", "linewidth": 0.5},
        capprops={"color": "black", "linewidth": 0.5},
    )
    #    clip = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

    # ax.set_ylim((0.01, 0.05))
    for i, ax in enumerate(ax):
        if dataset == "CIFAR10":
            ax.set_ylim((-0.001, 0.09))
        else:
            ax.set_ylim((-0.01, torch.maximum(ijepa.max(), clip.max())))
        ax.set_xticks(ticks, labels)
        ax.set_xlabel(r"\# Train Steps (batch size $=128)$")
        if not i:
            ax.set_ylabel(r"Mean MSE for all Images $(\lambda = 50)$")
            ax.set_title("I-JEPA")
        else:
            ax.set_title("CLIP")

    # fig.suptitle(dataset)
    fig.supxlabel("Comparing MSE for I-JEPA (left) and CLIP (right)")
    fig.savefig(out_file or "mse_over_time_boxplot.pdf")


mse_over_time()
