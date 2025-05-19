# Grounding DINO Model Interface

Grounding DINO is a powerful open-set object detection model that combines DINO with grounded pre-training. It enables precise object detection based on text prompts, making it highly versatile for various computer vision tasks. For more information, refer to the [Grounding DINO paper](https://arxiv.org/abs/2303.05499).

This repository contains the model interface for the [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) model in the Dataloop platform.

## Features

Grounding DINO offers several powerful capabilities:

* **Open-Set Object Detection**: Detect objects based on text descriptions without requiring specific training for each class
* **High Accuracy**: Achieves state-of-the-art performance on various benchmarks
* **Text-Guided Detection**: Use natural language to specify what objects to detect
* **Zero-Shot Capabilities**: Detect objects without fine-tuning on specific datasets

## Model Architecture

Grounding DINO combines several key components:
- A text backbone for processing text prompts
- An image backbone for processing visual features
- A feature enhancer
- Language-guided query selection
- A cross-modality decoder

## Acknowledgments

This project builds upon the open-source library [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), created by IDEA Research. Grounding DINO is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
