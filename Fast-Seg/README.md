# Incognito: Real time background segmentation for video-conferencing
Package for live background segmentation for CPU based video-conferencing.

## Repo Structure
**data** : Contains dataset handler, sample data, final compressed models and results

**experiments** : Base model and Efficient model training experiments

**model** : Model architecture definition

**optimization** : Quantization and Pruning optimization techniques

**utils** : Helper utilities

**evaluate.py** : Model evaluation script

**live_webcam.py** : Demonstrates live webcam segmentation using final trained model

**train.py** : Base trainer



## Setup
Clone repository
Docker container - (WIP)
```
WIP
```


## Run Inference
Live background segmentation on your webcam
```
python live_webcam.py
```


## Saved Final Models
|          Optimization                      | Model Graph    |        Model Weights      |
| ------------------------------------------ |:---------------------:| ----------------|
| UNet with ResNet-18 backbone (224*224)     |  <add-link>              |        <add-link>      |
| UNet with ResNet-18 backbone (128*128)                     |  <add-link>              |        <add-link>      |
| UNet with ResNet-34 backbone (224*224)                     |  <add-link>              |        <add-link>      |
| UNet with ResNet-34 backbone (128*128)                     |  <add-link>              |        <add-link>      |
| UNet with EfficientNet EB0 backbone (224*224)                     |  <add-link>             |        <add-link>      |
| UNet with EfficientNet EB0 backbone (128*128)                     |  <add-link>              |        <add-link>      |
| UNet with MobileNetV2 backbone (224*224)                     |  <add-link>              |        <add-link>      |
| UNet with MobileNetV2 backbone (128*128)                     |  <add-link>              |        <add-link>      |

## Analysis
- Quantization

```
#  Weight quantization using TF (fake quantization) results in model compression but not inference speedup.
```

- Pruning

```
# Weight Pruning
  Pruning weights of the NN results in sparse matrices. To get inference speedup from sparse matrices, you need specialized libraries like gemmlowp which can optimize matrix multiplications for sparse matrices

#Filter Pruning
  For complex model architectures like EfficientNets, it is difficult to rank Filters based on their importance

```
