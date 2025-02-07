from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from with_argparse import with_argparse

mpl.rcParams["text.usetex"] = True
DS_NAMES = {"imagenet": "ImageNet", "cifar10": "CIFAR10", "flickr30": "Flickr30K"}


@with_argparse
def fid_over_time(
    dataset: Literal["cifar10", "imagenet", "flickr30"],
    model: Literal["inceptionv3", "dinov2"] = "inceptionv3",
    time_is_train_time: bool = False,
):
    """
    Generate the plot for FID/FDD improving with the number of inference steps

    Author: Moritz
    """

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 3))
    cmap = mpl.colormaps["tab20c"].colors

    labels = [10, 25, 50, 75, 100]
    if model == "inceptionv3":
        if dataset == "cifar10":
            clip = torch.tensor([215.178, 209.557, 137.874, 75.456, 46.474])
            ijepa = torch.tensor([288.877, 223.054, 162.240, 97.881, 56.444])
        elif dataset == "imagenet":
            clip = torch.tensor(
                [
                    35.00402840878229,
                    29.92497773620704,
                    25.85683240706402,
                    24.101429289544456,
                    23.344319357348297,
                    24.01698646261167,
                    23.156319003575675,
                    22.464465777922044,
                ]
            )
            ijepa = torch.tensor(
                [
                    26.47907955176248,
                    20.124762187296085,
                    16.914779437759023,
                    15.531677249700863,
                    14.990195188937264,
                    14.85231876096043,
                    13.918775451683757,
                    13.728435742051772,
                ]
            )
            labels = [10, 20, 30, 40, 50, 60, 70, 80]
        elif dataset == "flickr30":
            clip = torch.tensor(
                [84.157, 49.832, 34.399, 28.476, 25.441, 22.755, 22.327]
            )
            ijepa = torch.tensor(
                [256.227, 63.253, 39.097, 29.979, 25.184, 20.009, 19.038]
            )
            labels = [10, 20, 30, 40, 50, 75, 100]
        else:
            raise NotImplementedError(dataset)
    elif model == "dinov2":
        if dataset == "imagenet":
            clip = torch.tensor(
                [
                    483.6327177186031,
                    398.37649185707915,
                    345.71003124376693,
                    319.4035108013868,
                    316.12718639328614,
                    315.63490494021016,
                    302.09806392134124,
                    298.62032606066896,
                ]
            )
            ijepa = torch.tensor(
                [
                    353.0907041265755,
                    272.19966455823305,
                    235.78038411127727,
                    221.99693286199363,
                    216.6956576528064,
                    213.94503570488132,
                    205.46539213718825,
                    201.5517595200431,
                ]
            )
            labels = [10, 20, 30, 40, 50, 60, 70, 80]
            labels = [f"{n}K" for n in labels]
            labels = labels[: clip.numel()]
        else:
            raise NotImplementedError(dataset)
    else:
        raise NotImplementedError(model)

    ticks = list(range(clip.numel()))

    ax.set_xticks(ticks, labels)
    ax.set_xlabel(
        r"\# Diffusion Steps" if not time_is_train_time else r"\# Training Steps"
    )
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
    ylabel = "FID" if model == "inceptionv3" else "FDD"
    ax.set_title(DS_NAMES.get(dataset, dataset) + rf" -- {ylabel}")

    ax.set_ylabel(ylabel)
    fig.legend(loc="upper right", bbox_to_anchor=(0, 0, 0.96, 0.92))
    fig.savefig(dataset + f"_{ylabel.lower()}.pdf")


fid_over_time()
