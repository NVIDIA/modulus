# A Diffusion model for a 2d Kármán vortex street about a fixed cylinder

This example uses a Pytorch implementation of DDPM which can be found [here](https://github.com/lucidrains/denoising-diffusion-pytorch/). 

## Problem overview

Turbulent flows are notoriously difficult to model. The structures involved can be found across a 
wide range of both temporal and spatial lengthscales, and the high degree of non-linearity as well as sensitivity to the initial conditions
make this an especially challenging problem. The dependence on dimension means that even when symmetry is present 
one cannot just simulate each 2d slice independently. The prohibitive computational cost of modern simulation methods (notably DNS, RANS and LES) makes this field ripe for
machine learning, especially generative probabilistic AI, since potential applications often only require the distribution of the resulting flow from stochastic
initial conditions. This example applies a modern DDPM implementation to a 2d Kármán vortex street about a fixed cylinder, and
is able to effectively capture the flow distribution.

## Dataset

Produced by Prof. Francesca Di Mare et. al. at the Ruhr University Bochum, the dataset consists of $` 100,000 `$ images 
produced by traditional computational fluid dynamics methods (LES).
The simulation ran with a total of $` 15 \times 10^6 `$ cells; the flow lasts approximately $` 1.45 `$s in real time.
The images are greyscale, so that the pixel colour represents the difference between the velocity at that point and the mean flow field.
The code provided here can also further process the images, removing excess white pixels and compressing to $` 512 \times 512 `$ before training.
The full dataset is available upon request from [ross@math.tu-berlin.de](mailto:ross@math.tu-berlin.de).

## Model overview and architecture

The model is an implementation of DDPM combined with a transformer and shadowed by an Exponential Moving Average (EMA) model.
Due to the large number of parameters and slow training times involved, the model is also built with parallelisation across multiple GPUs in mind,
and is supported out of the box.
A UNet with five downsampling layers, interspersed with attention and ResNet blocks, is used to represent the decoder. 

![Real image on the left, model on the right](../../../docs/img/diffusion_karman.png)

## Getting Started

The scripts provided include code for training, sampling and preprocessing the dataset.
For training, to view the available command line arguments, run

```bash
python main.py -h
```

and for sampling

```bash
python sample.py -h
```

Arguments can be provided on the command line or in ```config.json```. For training, the only required argument is ```experiment_name```. 
For sampling, ```model``` is also required. 
Providing this argument to the training script resumes training from where you left off.

Depending on your particular setup, you may need to set the environment variables
```bash
export WORLD_SIZE=$N
export CUDA_VISIBLE_DEVICES=$N
```
for some `$N` equivalent to the number of available GPUs.

## References
 - [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
 - [Generative Modelling of Turbulence](https://arxiv.org/abs/2112.02548)