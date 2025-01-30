from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from with_argparse import with_argparse

mpl.rcParams["text.usetex"] = True


@with_argparse
def mse_over_time(
    dataset: Literal["cifar10", "imagenet"],
):
    """
    Generate the plot for FID improving with the number of inference steps

    Author: Moritz
    """

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 3))
    cmap = mpl.colormaps["tab20c"].colors

    if dataset == "cifar10":
        clip = torch.tensor([215.178, 139.087, 98.795, 85.194])
        ijepa = torch.tensor([219.167, 151.526, 114.596, 92.782])
    elif dataset == "imagenet":
        clip = torch.tensor([161.554, 65.59, 24.993, 21.871])
        ijepa = torch.tensor([176.248, 98.082, 25.323, 14.825])
    else:
        raise NotImplementedError(dataset)

    ticks = list(range(4))
    labels = [10, 25, 50, 75]

    ax.set_xticks(ticks, labels)
    ax.set_xlabel(r"\# Diffusion Steps")
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
    ax.set_title("ImageNet" if dataset == "imagenet" else "CIFAR10")

    ax.set_ylabel("FID")
    fig.legend(loc="upper right", bbox_to_anchor=(0, 0, 0.96, 0.92))
    fig.savefig(dataset + "_fid.pdf")


mse_over_time()
