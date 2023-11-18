# Physics Informed DeepONet for Darcy flow

This example demonstrates physics informing of a data-driven model using the DeepONet
approach and the physics utilities from Modulus Sym.

## Problem overview

This is an extension of the 2D darcy flow data-driven problem. In addition to the
data loss, we will demonstrate the use of physics constranints, specifically
the equation residual loss. [Modulus Sym](https://github.com/NVIDIA/modulus-sym)
has utilities tailored for physics-informed machine learning. It also presents an
abstracted APIs that allows users to think and model the problem from the lens of
equations, constraints, etc. In this example, we will only levarage the physics-informed
utilites to see how we can add physics to an existing data-driven model with ease while
still maintaining the flexibility to define our own training loop and other details.
For a more abstracted definition of these type of problems, where the training loop definition
etc. is taken care of implictily, you may refer
[Modulus Sym](https://github.com/NVIDIA/modulus-sym)

## Dataset

The training and validation datasets for this example can be found on the [Fourier Neural
Operator Github page](https://github.com/neuraloperator/neuraloperator). The downloading
and pre-processing of the data can also be done by running the below set of commands:

```bash
pip install -r requirements.txt
python download_data.py
```

## Model overview and architecture

In this example we will use a Fourier Neural Operator (FNO) in the branch net
and use a fully-connected network for the trunk net. The input to the branch network
is the input permeability field and the input to the trunk network is the x, y coordinates.
The output of the model is the pressure field. Having the mapping between the pressure field
and the input x and y through a fully-differentiable network will allow us to compute
the gradients of the pressure field w.r.t input x and y through automatic differentiation
through Modulus sym utils.

In this example we will use the `PDE` class from Modulus-Sym to symbolically define the
PDEs. This is very convinient and most natural way to define these PDEs and allows us to
print the equations to check for correctness. This also abstracts out the complexity of
converting the equation into a pytorch representation.

## Getting Started

To get started with the example, simply run,

```bash
python 
physics_informed_deeponet.py
```

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://www.nature.com/articles/s42256-021-00302-5)
