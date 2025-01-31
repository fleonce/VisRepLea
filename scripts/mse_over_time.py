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
    Generate the plot for MSE improving over time comparing CLIP vs. I-JEPA

    Author: Moritz
    """

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 3), sharey=True)
    cmap = mpl.colormaps["tab20c"].colors

    ticks = list(range(len(ijepa_files)))
    labels = [f"{i * 10}K" for i in range(1, len(ticks))] + ["Post Train"]
    ijepa = torch.stack(
        [torch.load(file, weights_only=True) for file in ijepa_files]
    ).mean(dim=1)
    clip = torch.stack(
        [torch.load(file, weights_only=True) for file in clip_files]
    ).mean(dim=1)
    #    ijepa = np.array([0.0350, 0.0301, 0.0280, 0.0273, 0.0271, 0.0260, 0.0261, 0.0260])
    #    clip = np.array([0.0801, 0.0702, 0.0634, 0.0610, 0.0615, 0.0596, 0.0564, 0.0558])

    ylim = 0.01, 0.11
    if dataset == "CIFAR10":
        ylim = 0.001, 0.06
    ax.set_ylim(ylim)
    ax.set_xticks(ticks, labels)
    ax.set_xlabel(r"\# Train Steps (batch size $=128)$")
    ax.plot(
        ticks,
        ijepa,
        "*-",
        color="black",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=0.5,
        markerfacecolor=cmap[4],
        label="I-JEPA",
    )
    ax.plot(
        ticks,
        clip,
        "*-",
        color="black",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=0.5,
        markerfacecolor=cmap[12],
        label="CLIP",
    )
    ax.set_title(dataset)

    #    ax.set_ylabel(r'$\sigma (\lambda = 50)$')
    ax.set_ylabel(r"Mean MSE over all Images $(\lambda = 50)$")
    fig.legend(loc="upper right", bbox_to_anchor=(0, 0, 0.96, 0.92))
    fig.savefig(out_file or "mse_over_time.pdf")


mse_over_time()
