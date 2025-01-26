import os
from pathlib import Path

import torch
from datasets import load_from_disk
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToPILImage, ToTensor
from tqdm import tqdm

from visprak.metrics import StableDiffusionImageVariationPipeline
from visprak.training import test_collate_fn


def generate_images(
    unet_path: Path,
    dataset_path: Path,
    output_path: Path,
    inference_steps: int,
    diffusion_model: str = "CompVis/stable-diffusion-v1-4",
    batch_size: int = 64,
    dataloader_num_workers: int = 0,
    seed: int = 42,
    device: str = "cuda",
    resolution: int = 224,
):
    assert unet_path.exists()
    assert unet_path.is_dir()
    assert dataset_path.exists()
    assert dataset_path.is_dir()

    test_dataset = load_from_disk(dataset_path)["test"]

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

    pipeline = StableDiffusionImageVariationPipeline(
        vae=vae,
        image_encoder=None,
        unet=unet,
        feature_extractor=None,
        scheduler=noise_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    )

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

    images = list()
    orig_images = list()
    to_tensor = ToTensor()
    for batch in tqdm(
        test_dataloader, leave=False, desc="Diffusing validation images ..."
    ):
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

    images = torch.stack(images, dim=0)
    orig_images = torch.stack(orig_images, dim=0)

    to_pil = ToPILImage()
    pil_images = [to_pil(image) for image in images]
    orig_pil_images = [to_pil(image) for image in orig_images]

    os.makedirs(output_path, exist_ok=True)
    for i, (image, orig_image) in enumerate(
        zip(tqdm(pil_images, desc="Saving images", leave=False), orig_pil_images)
    ):
        image.save(os.path.join(output_path, f"{i:05d}_output.png"))
        orig_image.save(os.path.join(output_path, f"{i:05d}_target.png"))

    pass


generate_images()
