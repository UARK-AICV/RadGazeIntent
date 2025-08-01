# Configuration Guide

This directory contains configuration files for training and inference with RadGazeIntent.

## Training Configurations

### `train_for_real_egd_s2.json`
Configuration for training on EGD dataset with RadSeq (systematic sequential) intention labeling.

### `train_for_real_egd_s3.json` 
Configuration for training on EGD dataset with RadExplore intention labeling.

### `train_for_real_reflacx_s2.json`
Configuration for training on REFLACX dataset with RadSeq intention labeling.

## Inference Configurations

### `infer_egd_s2.json`
Configuration for inference on EGD test set.

## Backbone Configuration

### `resnet50.yaml`
ResNet-50 backbone configuration for feature extraction.

## Key Configuration Parameters

### Data Parameters
- `im_w`, `im_h`: Input image dimensions (224x224)
- `patch_num`: Number of patches for attention (14x14)
- `max_traj_length`: Maximum fixation sequence length (200)
- `fovea_radius`: Radius for foveal vision region
- `TAP`: Target-absent present setting ("TP" = target present)

### Model Parameters
- `hidden_dim`: Hidden dimension size (384)
- `num_layers_encoder`: Number of encoder layers (4) 
- `num_layers_decoder`: Number of decoder layers (6)
- `num_heads`: Number of attention heads (8)
- `dropout`: Dropout rate (0.1)

### Training Parameters
- `lr`: Learning rate (1e-5)
- `epochs`: Number of training epochs (100)
- `batch_size`: Training batch size (32)
- `eval_freq`: Evaluation frequency

## Customizing Configurations

To create custom configurations:

1. Copy an existing config file
2. Modify parameters as needed
3. Update dataset paths and settings
4. Adjust model architecture if required
5. Set appropriate training hyperparameters

## Dataset Path Configuration

Make sure to update the following paths in your config files:
- `dataset_root`: Root directory of your processed dataset
- `label_file`: JSON file containing fixation sequences and labels
- `backbone_config`: Path to backbone configuration file

Example:
```json
{
  "Data": {
    "label_file": "/path/to/your/dataset/labels.json",
    "backbone_config": "./configs/resnet50.yaml"
  }
}
```
