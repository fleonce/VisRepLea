import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from tqdm import tqdm
from with_argparse import with_argparse

GT_FORMAT = "{0}_target.png"
OUT_FORMAT = "{0}_output.png"

TRANSFORMS = transforms.Compose(
    [
        transforms.ToDtype(torch.uint8),
        transforms.ToImage(),
    ]
)
mpl.rcParams["text.usetex"] = True


@with_argparse(use_glob={"clip_outputs"})
def image_over_time_plot(
    ground_truth: Path,
    clip_outputs: list[Path],
    ijepa_outputs: list[Path],
    image_names: list[str],
    dataset: str = "ImageNet",
    figsize: list[float] = (4, 1.5),
    border_size: float = 0.05,
    y_border_size: float = 0.05,
    x_is_diffusion_steps: bool = False,
    out_filename: str = "image_over_time.pdf",
):
    """
    Convert lists of images to a plot over either training or diffusion time

    @author Moritz
    """
    assert ground_truth.is_dir()
    assert all(map(lambda x: x.is_dir(), clip_outputs)), clip_outputs
    assert all(map(lambda x: x.is_dir(), ijepa_outputs)), ijepa_outputs
    assert len(clip_outputs) == len(ijepa_outputs)

    def path_to_number(p: Path):
        if x_is_diffusion_steps:
            return int(p.name.split("-")[1])
        return int(p.parent.name[:-1])  # remove the K from a thousand steps

    clip_outputs.sort(key=path_to_number)
    ijepa_outputs.sort(key=path_to_number)
    clip_diffusion_steps = list(map(path_to_number, clip_outputs))

    gt_images = list()
    ijepa_images, clip_images = list(), list()
    for image_name in tqdm(image_names):
        gt_images.append(
            torch.from_numpy(
                np.asarray(Image.open(ground_truth / GT_FORMAT.format(image_name)))
            )
        )
        clip_images.append(load_img_from_time_paths(clip_outputs, image_name))
        ijepa_images.append(load_img_from_time_paths(ijepa_outputs, image_name))

    # stack and transpose color dim after pixel dims
    gt_images = torch.stack(gt_images)
    clip_images = torch.stack(clip_images)
    ijepa_images = torch.stack(ijepa_images)

    fig, ax = plt.subplots(
        2 * gt_images.size(0), clip_images.size(1) + 1, dpi=300, figsize=figsize
    )
    plt.subplots_adjust(
        wspace=border_size,
        hspace=border_size,
        left=border_size,
        bottom=1.5 * y_border_size,
        right=1 - border_size,
        top=1 - y_border_size,
    )
    #    fig.subplots_adjust(right=0.9)
    # fig.patch.set_linewidth(10)
    # fig.patch.set_edgecolor('cornflowerblue')
    for i, j in itertools.product(*map(range, ax.shape)):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

    for i in range(gt_images.size(0) * 2):
        # first, render the gt twice per image
        ax[i, -1].imshow(gt_images[i // 2])
        for j in range(clip_images.size(1)):
            if i % 2 == 0:
                ax[i, j].imshow(clip_images[i // 2, j])
            else:
                ax[i, j].imshow(ijepa_images[i // 2, j])

    for i in range(clip_images.size(1)):
        ax[-1, i].set_xlabel(
            r"\footnotesize{}"
            + str(clip_diffusion_steps[i])
            + ("K" if not x_is_diffusion_steps else "")
        )
    if not x_is_diffusion_steps:
        ax[-1, -2].set_xlabel(r"\footnotesize{}Post Train")
    for i in range(gt_images.size(0)):
        ax[i * 2, 0].set_ylabel(r"\footnotesize{}" + "CLIP", rotation=90)
        ax[i * 2 + 1, 0].set_ylabel(r"\footnotesize{}" + "I-JEPA", rotation=90)
    ax[-1, -1].set_xlabel(r"\footnotesize{}" + "Target")
    if x_is_diffusion_steps:
        fig.supxlabel(r"\footnotesize{}" + r"Diffusion Steps $\lambda$")
    else:
        fig.supxlabel(r"\footnotesize{}" + "Training Steps")
    fig.suptitle(dataset)
    fig.savefig(out_filename)


def load_img_from_time_paths(paths: list[Path], img_name: str) -> torch.Tensor:
    images = list()
    for path in paths:
        images.append(
            torch.from_numpy(np.asarray(Image.open(path / OUT_FORMAT.format(img_name))))
        )
    return torch.stack(images)


image_over_time_plot()
