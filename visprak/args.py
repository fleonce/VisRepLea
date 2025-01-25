from dataclasses import dataclass
from typing import Literal, Optional

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


@dataclass
class VisRepLeaArgs:
    """
    Arguments setup for the "Praktikum Visual Representation Learning"
    assignment on comparing richness of information in CLIP and I-JEPA embeddings
    """

    diffusion_model: str = "CompVis/stable-diffusion-v1-4"

    cross_attention_dim: int = 768
    input_perturbation: float = 0
    revision: str = None
    variant: str = None
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    output_dir: str = "visreplea_models"
    logging_dir: str = "logs"
    image_logging_dir: str = "images"
    cache_dir: str = None
    seed: int = 42
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    gradient_checkpointing: bool = False
    train_batch_size: int = 16
    test_batch_size: int = 16
    num_train_epochs: int = 100
    max_train_steps: int | None = None
    train_test_interval: float = 0.1
    train_test_fraction: float = (
        0.05  # during training, validate just 5% of the testing data
    )
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant_with_warmup",
    ] = "constant"
    lr_warmup_steps: int = 500
    snr_gamma: float | None = None
    dream_training: bool = False
    dream_detail_preservation: float = 1.0
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    use_ema: bool = False
    offload_ema: bool = False
    foreach_ema: bool = False
    non_ema_revision: str | None = None
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    hub_token: str | None = None
    prediction_type: Optional[Literal["epsilon", "v_prediction"]] = None
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"
    report_to: Literal["tensorboard", "wandb", "comet_ml", "all"] = "tensorboard"
    local_rank: int = -1
    checkpointing_steps: int = 500
    checkpoints_total_limit: int | None = None
    resume_from_checkpoint: str | None = None
    enable_xformers_memory_efficient_attention: bool = False
    noise_offset: float = 0.0
    validation_epochs: int = 5
    tracker_project_name: str = "visreplea"
    train_data_dir: str = "training_data"
    image_column: Optional[str] = None
