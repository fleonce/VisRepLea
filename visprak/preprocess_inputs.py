import os.path
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from transformers import CLIPVisionModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from with_argparse import with_dataclass

from visprak.utils import DATASET_NAME_MAPPING, DATASET_URL_MAPPING


@dataclass
class PreprocessArgs:
    embedding_model: Literal["clip", "i-jepa"]
    dataset: Literal["cifar10", "imagenet"]
    data_dir: str
    embedding_device: str = "cuda"
    batch_size: int = 128
    embedding_model_path: str = "openai/clip-vit-base-patch16"
    cache_dir: str = None
    image_column: str | None = None
    resolution: int = 224
    center_crop: bool = False
    random_flip: bool = False
    shard_size: int = 5000
    seed: int = 42
    max_train_samples: int | None = None
    max_test_samples: int | None = None


@with_dataclass(dataclass=PreprocessArgs)
def preprocess_inputs(args: PreprocessArgs):
    if args.dataset in {"cifar10", "imagenet"}:
        dataset_url = DATASET_URL_MAPPING.get(args.dataset, None)
        dataset = load_dataset(
            dataset_url,
            cache_dir=args.cache_dir,
            # data_dir=args.data_dir,
        )
    else:
        raise NotImplementedError(args.dataset)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset, None)
    if args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, False),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, False),
        ]
    )

    embedding_transforms = transforms.Compose(
        [
            # [0..255] -> [0:1]
            transforms.ToDtype(torch.float32, True),
            transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
        ]
    )

    if args.embedding_model == "clip":
        image_model = CLIPVisionModel.from_pretrained(args.embedding_model_path).to(
            args.embedding_device
        )
    elif args.embedding_model == "i-jepa":
        image_model = ...
    else:
        raise NotImplementedError(args.embedding_model)

    def preprocess_fn(examples, is_train: bool):
        images = [image.convert("RGB") for image in examples[image_column]]
        transform_fn = train_transforms if is_train else test_transforms
        examples["pixel_values"] = torch.stack(
            [transform_fn(image) for image in images]
        )
        return examples

    if args.max_train_samples is not None:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=args.seed)
            .select(range(args.max_train_samples))
        )
    if args.max_test_samples is not None:
        dataset["test"] = (
            dataset["test"].shuffle(seed=args.seed).select(range(args.max_test_samples))
        )

    test_dataset = (
        dataset["test"]
        .map(
            partial(preprocess_fn, is_train=False),
            batched=True,
            remove_columns=column_names,
        )
        .with_format("torch")
    )
    train_dataset = (
        dataset["train"]
        .map(
            partial(preprocess_fn, is_train=True),
            batched=True,
            remove_columns=column_names,
        )
        .with_format("torch")
    )

    def embedding_fn(examples):
        with torch.no_grad():
            inputs = embedding_transforms(examples["pixel_values"]).to(
                args.embedding_device
            )
            outputs = image_model(inputs)[0]
        examples["latent"] = outputs
        return examples

    test_dataset = test_dataset.map(
        embedding_fn, batched=True, batch_size=args.batch_size
    )
    train_dataset = train_dataset.map(
        embedding_fn, batched=True, batch_size=args.batch_size
    )
    train_dataset.save_to_disk(os.path.join(args.data_dir, "train"))
    test_dataset.save_to_disk(os.path.join(args.data_dir, "test"))


if __name__ == "__main__":
    preprocess_inputs()
