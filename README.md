# Satellite Image Classification using Vision Transformer (ViT)

## Overview

This project focuses on fine-tuning a pretrained Vision Transformer (ViT) model for satellite image classification using the EuroSAT dataset.

The model is trained to classify land-use categories from multispectral satellite images.

## Dataset

* EuroSAT dataset
* ~27,000 images
* 10 land-use classes (e.g., residential, forest, river, highway)

## Model

* Pretrained: google/vit-base-patch16-224-in21k
* Fine-tuned for classification task

## Pipeline

* Data preprocessing and normalization
* Patch embedding (ViT input processing)
* Multi-head self-attention mechanism
* Classification head fine-tuning
* Model evaluation and inference

## Results

* Test Accuracy: **98.85%**

## Tech Stack

* Python
* PyTorch
* HuggingFace Transformers

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Open the notebook:
   jupyter notebook vit_eurosat_classification.ipynb

## Future Improvements

* Deploy as a web app
* Try larger ViT models
* Experiment with data augmentation

## Author

Sashank Talluri
