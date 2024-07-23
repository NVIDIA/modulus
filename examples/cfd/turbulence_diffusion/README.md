# Diffusion for a 2d Kármán vortex street about a fixed cylinder

This example uses a Pytorch implementation of DDPM (package link) 

## Problem overview

Problem Overview

## Dataset

Produced by Prof. Francesca Di Mare et. al. at the University of Bochum, the dataset consists of $` 100,000 `$ images 
produced by traditional computational fluid dynamics methods (LES).
The simulation ran with a total of $` 15 \times 10^6 `$ cells; the flow lasts approximately $` 1.45 `$s in real time.
The images are greyscale, so that the pixel colour represents the difference between the velocity at that point the mean flow field.
The code provided here can also further process the images, removing excess white pixels and compressing to $` 512 \times 512 `$ before training.
The full dataset is available upon request from [ross@math.tu-berlin.de](mailto:ross@math.tu-berlin.de).

## Model overview and architecture

The model is an implementation of DDPM combined with a transformer and shadowed by an Exponential Moving Average (EMA) model.
Due to the large number of parameters and slow training times involved, the model is also built with parallelisation across multiple GPUs in mind,
and is supported out of the box.
A UNet with five downsampling layers, interspersed with attention and ResNet blocks, is used to represent the decoder. 

## Getting Started

The scripts provided include code for training, sampling and preprocessing the dataset.
For training, to view the available command line arguments, run

```bash
python train.py -h
```

and for sampling

```bash
python sample.py -h
```

For training, the only required argument is ```experiment_name```. 
For sampling, ```model``` is also required. 
Providing this argument to the training script resumes training from where you left off.

Depending on your particular setup, you may need to set the environment variables
```bash
export WORLD_SIZE=$N
export CUDA_VISIBLE_DEVICES=$N
```
for some `$N` equivalent to the number of available GPUs.

## References //TODO:
 - DDPM Pytorch Implementation
 - Exponential Moving average
 - Di Maare Dataset
 - Claudia GAN paper