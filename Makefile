mse_plot:
	venv/bin/python3 scripts/mse_over_time.py --ijepa_file over_time/tf-imagenet-ijepa/*/*.bin --clip_file over_time/tf-imagenet-clip/*/*.bin

COMMA := ,
CLIP_MODELS := openai/clip-vit-large-patch14
CLIP_MODELS += openai/clip-vit-base-patch32
CLIP_MODELS += openai/clip-vit-large-patch14-336
CLIP_MODELS += openai/clip-vit-base-patch16
CLIP_MODEL_SIZES := $(foreach model,$(CLIP_MODELS),$(model)$(COMMA)--model_class$(COMMA)transformers.CLIPVisionModel$(COMMA)--config_class$(COMMA)transformers.CLIPVisionConfig)
JEPA_MODELS := facebook/ijepa_vith14_1k
JEPA_MODELS += facebook/ijepa_vith14_22k
JEPA_MODELS += facebook/ijepa_vith16_1k
JEPA_MODELS += facebook/ijepa_vitg16_22k
JEPA_MODEL_SIZES := $(foreach model,$(JEPA_MODELS),$(model)$(COMMA)--model_class$(COMMA)transformers.IJepaModel$(COMMA)--config_class$(COMMA)transformers.IJepaConfig)

MODEL_SIZES := $(CLIP_MODEL_SIZES) $(JEPA_MODEL_SIZES)

model_sizes: $(MODEL_SIZES);

$(MODEL_SIZES):
	@python3 scripts/get_model_size.py --pretrained_model_name_or_path $(subst $(COMMA), ,$@)

.PHONY: model_sizes mse_plot

image_over_time_flickr.pdf: scripts/figures/image_over_time_plot.py
	python3 $< --x_is_diffusion_steps --ground_truth tgt-flickr/ --clip_output sd-flickr/clip/inference_steps-* --ijepa_output sd-flickr/ijepa/inference_steps-* --image_name 00101 00007 00125 00286 00242 00241 --figsize 5 7.5 --dataset Flickr30K --out_filename $@

image_over_time_private.pdf: scripts/figures/image_over_time_plot.py
	@python3 $< --x_is_diffusion_steps --ground_truth sd-memes/clip/inference_steps-10/ --clip_output sd-memes/clip/inference_steps-* --ijepa_output sd-memes/ijepa/inference_steps-* --image_name 00000 00001 00002 00003 00004 --figsize 5 6.25 --dataset "Private Collection" --out_filename $@

image_over_train_time_cifar10.pdf: scripts/figures/image_over_time_plot.py
	@python3 $< --ground_truth tgt-cifar10 --clip_output tf-cifar10/clip/*/inference_steps-50/ --ijepa_output tf-cifar10/ijepa/*/inference_steps-50/ --image_name 00015 00035 00091 00021 00053 --border_size 0.075 --y_border_size 0.05 --figsize 3 6 --dataset "CIFAR10" --out_filename $@

clean:
	rm image_over_*.pdf

.PHONY: image_over_train_time_cifar10.pdf image_over_time_private.pdf image_over_time_flickr.pdf
