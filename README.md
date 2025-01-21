# CausalSR

A PyTorch implementation of "CausalSR: Structural Causal Model-Driven Super-Resolution with Counterfactual Inference"

## Overview

CausalSR introduces a novel approach to image super-resolution by incorporating structural causal models and counterfactual inference. The model explicitly models degradation mechanisms through causal relationships, leading to improved restoration quality especially for complex real-world scenarios.

## Features

- Structural causal modeling of image degradation
- Semantic-guided restoration using CLIP features  
- Counterfactual learning framework
- Adaptive intervention mechanism
- Support for multiple degradation types (blur, noise, JPEG artifacts)

## Requirements

```python
torch>=1.9.0
torchvision>=0.10.0 
numpy>=1.19.2
opencv-python>=4.5.3
pillow>=8.3.1
clip @ git+https://github.com/openai/CLIP.git

## Training

python train.py --config configs/train_config.yaml

## Testing

python test.py --model path/to/checkpoint.pth --input path/to/image.png
