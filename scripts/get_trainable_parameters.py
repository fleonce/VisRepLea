import torch
from diffusers import UNet2DConditionModel

from visprak.utils import freeze_unet_except_for_cross_attn
from with_argparse import with_argparse


@with_argparse
def get_trainable_parameters(
    unet_pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
    subfolder: str | None = "unet",
    cross_attention_dim: int = 1024,
):
    with torch.device("meta"):
        unet_config = UNet2DConditionModel.load_config(
            unet_pretrained_model_name_or_path, subfolder=subfolder
        )
        unet_config["cross_attention_dim"] = cross_attention_dim  # noqa
        unet = UNet2DConditionModel(**unet_config)
        freeze_unet_except_for_cross_attn(unet)
        num_trainable = 0
        num_parameters = 0
        for name, param in unet.named_parameters():
            if param.requires_grad:
                num_trainable += 1
                num_parameters += param.numel()
                print(name, param.shape)
        print("num_trainable", num_trainable)
        print("num_parameters", num_parameters, "%.0f M" % (num_parameters / 1e6,))


get_trainable_parameters()
