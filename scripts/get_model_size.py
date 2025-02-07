import importlib

import torch
from with_argparse import with_argparse


@with_argparse
def get_model_size(
    pretrained_model_name_or_path: str, config_class: str, model_class: str
) -> None:
    """
    @param pretrained_model_name_or_path: The Huggingface checkpoint name available online or as a file
    @param model_class: the class to initialize the model from
    @param config_class: the class to initialize the config from
    @author Moritz
    """
    if "." not in model_class:
        print("Specify whole config class")
        exit(1)
    pkg, name = config_class.rsplit(".", 1)
    config_package = importlib.import_module(pkg)
    config_class = getattr(config_package, name)
    config = config_class.from_pretrained(pretrained_model_name_or_path)

    pkg, name = model_class.rsplit(".", 1)
    model_package = importlib.import_module(pkg)
    model_class = getattr(model_package, name)
    with torch.device("meta"):
        model = model_class(config)
        print(pretrained_model_name_or_path)
        print(
            pretrained_model_name_or_path,
            "numel parameters",
            sum(map(lambda x: 1, model.parameters())),
        )
        print(
            pretrained_model_name_or_path,
            "parameters",
            f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M",
            sum(p.numel() for p in model.parameters()),
        )
        print(pretrained_model_name_or_path, "hidden_size", config.hidden_size)
        print(
            pretrained_model_name_or_path, "num_hidden_layers", config.num_hidden_layers
        )
        print(
            pretrained_model_name_or_path,
            "num_attention_heads",
            config.num_attention_heads,
        )


get_model_size()
