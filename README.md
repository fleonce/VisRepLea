# VisRepLea

### Getting started

To install this package, run
```bash
pip install git+https://github.com/fleonce/VisRepLea
```
which will install all required dependencies.

There are two git submodules for FID and FDD, available in `external/pytorch_fid` and `external/fdd` respectively.

### Repository Structure

```text
VisRepLea/
├─ external/    # git submodules
├─ scripts/     # scripts for creating plots
|   ├─ figures/
|   |   ├─ image_over_time_plot.py  # visualize image improving over (training/diffusion) time
|   ├─ compare_model_size.py        # compare clip + ijepa by parameters
|   ├─ fid_over_inference_steps.py  # fid plot
|   ├─ get_model_size.py            # get model size of any model
|   ├─ get_trainable_parameters.py  # get trainable params of UNet2D in our setting
|   ├─ mse_over_time.py             # use results from `visprak/mean_error.py` to create barplots
|   ├─ mse_over_time_boxplot.py     # use results from `visprak/mean_error.py` to create box plots
|   ├─ prepare_flickr.py            # create flickr in our dataset format
├─ visprak/     # preprocessing & training
|   ├─ args.py                      # arguments for the main training
|   ├─ custom_dataset.py            # create a custom dataset
|   ├─ generate_images.py           # generate images of given diffusion steps with a checkpoint
|   ├─ mean_error.py                # calculate the MMSE metric for a folder of images
|   ├─ metrics.py                   # saving images during training
|   ├─ pipeline.py                  # custom Stable Diffusion Pipeline to accept preprocessed latents
|   ├─ preprocess_inputs.py         # preprocessing script
|   ├─ show_images.py               # generate images from preprocessed datasets -> sanity checking
|   ├─ training.py                  # main training implementation
|   ├─ utils.py                     # utilities such as parameter freezing
├─ Makefile     # quickly get model size of any CLIP/I-JEPA model
|               #  via `make model_sizes`
```

### CLI Reference:

##### Creating Custom Datasets

If dataset is not available on Huggingface, use this to create one from images.
```shell
python3 visprak/custom_dataset.py
  --train_directory path/to/train_pngs
  --test_directory path/to/test_pngs
  --save_path path/to/save/dir
```

##### Preprocessing Inputs

When the dataset is ready, preprocess it for both models via:
```shell
python3 visprak/preprocess_inputs.py
  --embedding_model {clip,i-jepa}
  --embedding_model_path {huggingface_model_name_or_path}
  --dataset huggingface_dataset
  --data_dir path/to/save/dir
  --batch_size 128    # {batch size for embedding}
  --resolution 224    # target resolution
  --seed 42           # seed for limiting dataset size
  --max_train_samples # limit train size
  --max_test_samples  # limit test size
```

##### Training

Finally, perform training:
```shell
python3 visprak/training.py
  --diffusion_model         # diffusion model path
  --cross_attention_dim     # dim of the latents that are passed to SD
  --train_data_dir          # `--data_dir` of preprocess_inputs.py
  --train_batch_size        # batch size
  --mixed_precision         # use bf16 maybe?
  --{lots of more options available via -h}
```
