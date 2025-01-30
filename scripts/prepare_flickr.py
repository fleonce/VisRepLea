from pathlib import Path

from datasets import load_dataset
from with_argparse import with_argparse


@with_argparse
def prepare_flickr(save_path: Path):
    dataset = load_dataset("nlphuji/flickr30k")
    dataset = dataset.filter(lambda x: x["split"] == "test")
    print(len(dataset))
    dataset.save_to_disk(save_path)


prepare_flickr()
