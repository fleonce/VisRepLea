import os
from pathlib import Path

from datasets import Dataset, DatasetDict

from PIL import Image
from with_argparse import with_argparse


@with_argparse
def custom_dataset(
    train_directory: Path,
    test_directory: Path,
    save_path: Path,
):
    dataset = DatasetDict(
        {
            "train": dataset_from_dir(train_directory),
            "test": dataset_from_dir(test_directory),
        }
    )
    dataset.save_to_disk(save_path)


def dataset_from_dir(directory: Path):
    images = [
        Image.open(directory / file)
        for file in os.listdir(directory)
        if file.endswith(".png")
    ]
    return Dataset.from_dict({"images": images})


custom_dataset()
