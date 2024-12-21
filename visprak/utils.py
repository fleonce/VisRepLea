from diffusers import UNet2DConditionModel


DATASET_URL_MAPPING = {
    "cifar10": "uoft-cs/cifar10",
    "imagenet": "ILSVRC/imagenet-1k",
}
DATASET_NAME_MAPPING = {
    "ILSVRC/imagenet-1k": ("image",),
    "uoft-cs/cifar10": ("img",),
}


def freeze_unet_except_for_cross_attn(unet: UNet2DConditionModel):
    for name, param in unet.named_parameters():
        if "attention" in name:
            param.requires_grad_(False)
