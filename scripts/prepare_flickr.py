from pathlib import Path

from datasets import load_dataset
from with_argparse import with_argparse


@with_argparse
def prepare_flickr(save_path: Path):
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    dataset = dataset.filter(lambda x: x["split"] == "test")
    dataset.save_to_disk(save_path)


prepare_flickr()
