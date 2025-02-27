import os
from pathlib import Path

import torch
from datasets import load_from_disk
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToPILImage, ToTensor
from tqdm import tqdm
from with_argparse import with_argparse

from visprak.pipeline import StableDiffusionPreprocessedImagesPipeline
from visprak.training import test_collate_fn


@with_argparse
def generate_images(
    unet_path: Path,
    dataset_path: Path,
    output_path: Path,
    inference_steps: list[int],
    diffusion_model: str = "CompVis/stable-diffusion-v1-4",
    batch_size: int = 64,
    dataloader_num_workers: int = 0,
    seed: int = 42,
    device: str = "cuda",
    resolution: int = 224,
    num_batches: int = 0,
):
    assert unet_path.exists()
    assert unet_path.is_dir()
    assert dataset_path.exists()
    assert dataset_path.is_dir()

    test_dataset = load_from_disk(dataset_path)

    vae = AutoencoderKL.from_pretrained(
        diffusion_model,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        unet_path,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        diffusion_model, subfolder="scheduler"
    )

    pipeline = StableDiffusionPreprocessedImagesPipeline(
        vae=vae,
        image_encoder=None,
        unet=unet,
        feature_extractor=None,
        scheduler=noise_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
    )

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    for inference_steps in inference_steps:
        generate_and_save_images(
            output_path / f"inference_steps-{inference_steps}",
            inference_steps,
            generator,
            test_dataloader,
            pipeline,
            device,
            resolution,
            num_batches,
        )


def generate_and_save_images(
    output_path: Path,
    inference_steps: int,
    generator: torch.Generator,
    test_dataloader: torch.utils.data.DataLoader,
    pipeline: StableDiffusionPreprocessedImagesPipeline,
    device: str | torch.device,
    resolution: int,
    num_batches: int,
):
    images = list()
    orig_images = list()
    to_tensor = ToTensor()
    for i, batch in enumerate(
        tqdm(
            test_dataloader,
            leave=False,
            desc="Diffusing validation images ...",
            total=num_batches or len(test_dataloader),
        )
    ):
        if i >= num_batches and num_batches:
            break
        try:
            with torch.autocast("cuda", torch.bfloat16):
                generation = pipeline(
                    batch["latent"].to(device),
                    num_inference_steps=inference_steps,
                    generator=generator,
                    width=resolution,
                    height=resolution,
                )
            orig_images.extend(batch["sd_images"].unbind())
            images.extend(to_tensor(generation.images))
        except KeyboardInterrupt:
            break

    images = torch.stack(images, dim=0)
    orig_images = torch.stack(orig_images, dim=0)

    to_pil = ToPILImage()
    pil_images = [to_pil(image) for image in images]
    orig_pil_images = [to_pil(image) for image in orig_images]

    os.makedirs(output_path, exist_ok=True)
    print("Saving images to", output_path.as_posix())
    for i, (image, orig_image) in enumerate(
        zip(tqdm(pil_images, desc="Saving images", leave=False), orig_pil_images)
    ):
        image.save(os.path.join(output_path, f"{i:05d}_output.png"))
        orig_image.save(os.path.join(output_path, f"{i:05d}_target.png"))


generate_images()
