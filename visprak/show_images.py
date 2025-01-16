import os

import torch
import torchvision.transforms.v2 as transforms
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToPILImage
from tqdm import tqdm
from with_argparse import with_dataclass

from visprak.preprocess_inputs import PreprocessArgs


@with_dataclass(dataclass=PreprocessArgs)
def main(args: PreprocessArgs):
    dataset = {
        "train": (
            load_from_disk(
                os.path.join(args.data_dir, "train"),
            )
        ),
        "test": (
            load_from_disk(
                os.path.join(args.data_dir, "test"),
            ).with_format("torch")
        ),
    }

    test_transforms = transforms.Compose(
        [
            transforms.ToDtype(torch.uint8),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    def test_collate_fn(examples):
        pixel_values = [
            test_transforms(example["pixel_values"]) for example in examples
        ]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"sd_images": pixel_values}

    # Create a test DataLoader as well to use during training (just for some validation) as well as after (full run)
    test_dataloader = DataLoader(
        dataset["test"],
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=64,
        num_workers=0,
    )

    orig_images = list()
    for batch in tqdm(
        test_dataloader, leave=False, desc="Diffusing validation images ..."
    ):
        orig_images.extend(batch["sd_images"].unbind())
    orig_images = torch.stack(orig_images, dim=0)
    to_pil = ToPILImage("RGB")
    orig_pil_images = [to_pil(image) for image in orig_images]

    save_dir = os.path.join(
        args.data_dir,
        "images",
        "target",
    )
    os.makedirs(save_dir, exist_ok=True)
    for i, (orig_image) in enumerate(orig_pil_images):
        orig_image.save(os.path.join(save_dir, f"{i:05d}_target.png"))


main()
