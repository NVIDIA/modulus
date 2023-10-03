# AeroGraphNet for external aerodynamic evaluation

This example demonstrates how to train the AeroGraphNet model for external aerodynamic
analysis of simplified (Ahmed body-type) car geometries. AeroGraphNet is based on the
MeshGraphNet architecture. It achieves good accuracy on predicting the pressure and
wall shear stresses on the surface mesh of the Ahmed body-type geometries, as well as
the drag coefficient.

## Problem overview

To goal is to develop an AI surrogate model that can use simulation data to learn the
external aerodynamic flow over parameterized Ahmed body shape. This serves as a baseline
for more refined models for realistic car geometries. The trained model can be used to
predict the change in drag coefficient,and surface pressure and wall shear stresses due
to changes in the car geometry. This is a stepping stone to applying similar approaches
to other application areas such as aerodynamic analysis of aircraft wings, real car
geometries, etc.

## Dataset

Industry-standard Ahmed-body geometries are characterized by six design parameters:
length, width, height, ground clearance, slant angle, and fillet radius. Refer
to the [wiki](https://www.cfd-online.com/Wiki/Ahmed_body) for details on Ahmed
body geometry. In addition to these design parameters, we include the inlet velocity to
address a wide variation in Reynolds number. We identify the design points using the
Latin hypercube sampling scheme for space filling design of experiments and generate
around 500 design points.

The aerodynamic simulations were performed using the GPU-accelerated OpenFOAM solver
for steady-state analysis, applying the SST K-omega turbulence model. These simulations
consist of 7.2 million mesh points on average, but we use the surface mesh as the input
to training which is roughly around 70k mesh nodes.

To request access to the full dataset, please reach out to the
[NVIDIA Modulus team](modulus-team@nvidia.com).

## Model overview and architecture

The AeroGraphNet model is based on the MeshGraphNet architecture which is instrumental
for learning from mesh-based data using GNNs. The inputs to the model are:

- Ahmed body surface mesh
- Reynolds number
- Geometry parameters (optional, including length, width, height, ground clearance,
slant angle, and fillet radius)
- surface normals (optional)

Output of the model are:

- Surface pressure
- Wall shear stresses
- Drag coefficient

![Comparison between the AeroGraphNet prediction and the
ground truth for surface pressure, wall shear stresses, and the drag coefficient for one
of the samples from the test dataset.](../../../docs/img/ahmed_body_results.png)

The input to the model is in form of a `.vtp` file and is then converted to
bi-directional DGL graphs in the dataloader. The final results are also written in the
form of `.vtp` files in the inference code. A hidden dimensionality of 256 is used in
the encoder, processor, and decoder. The encoder and decoder consist of two hidden
layers, and the processor includes 15 message passing layers. Batch size per GPU is
set to 1. Summation aggregation is used in the
processor for message aggregation. A learning rate of 0.0001 is used, decaying
exponentially with a rate of 0.99985. Training is performed on 8 NVIDIA A100
GPUs, leveraging data parallelism. Total training time is 4 hours, and training is
performed for 500 epochs.

## Getting Started

The dataset for this example is not publicly available. To get access, please reach out
to the [NVIDIA Modulus team](modulus-team@nvidia.com).

This example requires the `pyvista` and `vtk` libraries. Install with

```bash
pip install pyvista vtk
```

To train the model, run

```bash
python train.py
```

Data parallelism is also supported with multi-GPU runs. To launch a multi-GPU training,
run

```bash
mpirun -np <num_GPUs> python train.py
```

If running in a docker container, you may need to include the `--allow-run-as-root` in
the multi-GPU run command.

Progress and loss logs can be monitored using Weights & Biases. To activate that,
set `wandb_mode` to `online` in the `constants.py`. This requires to have an active
Weights & Biases account. You also need to provide your API key. There are multiple ways
for providing the API key but you can simply export it as an environment variable

```bash
export WANDB_API_KEY=<your_api_key>
```

The URL to the dashboard will be displayed in the terminal after the run is launched.
Alternatively, the logging utility in `train.py` can be switched to MLFlow.

Once the model is trained, run

```bash
python inference.py
```

This will save the predictions for the test dataset in `.vtp` format in the `results`
directory. Use Paraview to open and explore the results.

## References

- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
