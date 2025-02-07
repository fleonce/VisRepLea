import math
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.nn.functional import mse_loss
from tqdm import trange
from with_argparse import with_argparse


@torch.no_grad()
@with_argparse(use_glob={"directories"})
def mean_error(
    directories: list[Path],
    n_images: int,
    batch_size: int,
    target_directory: Optional[Path] = None,
):
    """
    Compute the mean squared error between pairs of images, given a directory and the number of pairs inside

    Args:
        directories (Path): The directory where the files are located.
            The images must be named the following way: `%05d_output.png` and `%05d_target.png`
        target_directory (Path): The directory where the target files are located.
        n_images (int): The number of images that should be compared.
        batch_size (int): The batch size for MSE calculation.

    Author: Moritz
    """
    for directory in directories:
        _mean_error(
            directory, n_images, batch_size, target_directory
        )

def _mean_error(
    directory: Path,
    n_images: int,
    batch_size: int,
    target_directory: Optional[Path] = None
):
    print(directory.as_posix())
    target_directory = target_directory or directory

    input_transforms = transforms.Compose(
        (transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True))
    )

    outputs = list()
    targets = list()
    for i in trange(n_images):
        outputs.append(input_transforms(Image.open(directory / f"{i:05d}_output.png")))
        targets.append(input_transforms(Image.open(target_directory / f"{i:05d}_target.png")))

    output_tensors = torch.stack(outputs)
    target_tensors = torch.stack(targets)

    print("min, max", *output_tensors[0].aminmax())

    n_slices = math.ceil(n_images / batch_size)
    mse_sum = 0
    mse_vec = None
    for output_batch, target_batch in zip(
        torch.tensor_split(output_tensors, n_slices, dim=0),
        torch.tensor_split(target_tensors, n_slices, dim=0),
    ):
        mean = mse_loss(output_batch, target_batch, reduction="none").mean(
            dim=(1, 2, 3)
        )
        if mse_vec is None:
            mse_vec = mean
        else:
            mse_vec = torch.cat((mse_vec, mean), dim=0)
        mse_sum += mean.sum()

    torch.save(mse_vec, directory.parent / "mse.bin")
    print("mean mse", mse_sum / n_images)


mean_error()
