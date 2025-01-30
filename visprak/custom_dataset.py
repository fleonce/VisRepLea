import os
from pathlib import Path

from datasets import Dataset, DatasetDict

from PIL import Image
from with_argparse import with_argparse


@with_argparse
def custom_dataset(
    in_directory: Path,
    save_path: Path,
):
    images = [
        Image.open(in_directory / file)
        for file in os.listdir(in_directory)
        if file.endswith(".png")
    ]
    test_dataset = Dataset.from_dict({"images": images})

    dataset = DatasetDict(
        {
            "train": test_dataset,
            "test": test_dataset,
        }
    )
    dataset.save_to_disk(save_path)


custom_dataset()
