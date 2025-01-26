import os.path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToPILImage, ToTensor
from tqdm import tqdm

from visprak.args import VisRepLeaArgs
from visprak.pipeline import StableDiffusionPreprocessedImagesPipeline


def log_validation(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    args: VisRepLeaArgs,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    global_step: int,
    test_dataloader: DataLoader,
    save_model: bool,
):
    """
    Author: Moritz

    derived from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
    """
    pipeline = StableDiffusionPreprocessedImagesPipeline(
        vae=vae,
        image_encoder=None,
        unet=unet,
        feature_extractor=None,
        scheduler=noise_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if save_model:
        pipeline.save_pretrained(args.output_dir)
    pipeline = pipeline.to(accelerator.device)
    pipeline.torch_dtype = weight_dtype
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = list()
    orig_images = list()
    to_tensor = ToTensor()
    for batch in tqdm(
        test_dataloader, leave=False, desc="Diffusing validation images ..."
    ):
        with torch.autocast("cuda", weight_dtype):
            generation: StableDiffusionPipelineOutput
            generation = pipeline(
                batch["latent"].to(accelerator.device),
                num_inference_steps=10,
                generator=generator,
                width=args.resolution,
                height=args.resolution,
            )
        orig_images.extend(batch["sd_images"].unbind())
        images.extend(to_tensor(generation.images))

    images = torch.stack(images, dim=0)
    orig_images = torch.stack(orig_images, dim=0)
    mse = F.mse_loss(orig_images, images, reduction="none").mean(dim=(1, 2, 3)).mean()

    to_pil = ToPILImage()
    pil_images = [to_pil(image) for image in images]
    orig_pil_images = [to_pil(image) for image in orig_images]

    save_dir = os.path.join(
        args.output_dir,
        args.image_logging_dir,
        f"global_step_{'final' if save_model else global_step}",
    )
    os.makedirs(save_dir, exist_ok=True)
    for i, (image, orig_image) in enumerate(
        zip(tqdm(pil_images, desc="Saving images", leave=False), orig_pil_images)
    ):
        image.save(os.path.join(save_dir, f"{i:05d}_output.png"))
        orig_image.save(os.path.join(save_dir, f"{i:05d}_target.png"))

    for tracker in accelerator.trackers:
        tracker.log(
            {
                "mse": mse.item(),
            },
            step=global_step,
        )
        tracker.log_images(
            {"images": images, "originals": orig_images}, step=global_step
        )
    pass


# sourced from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
