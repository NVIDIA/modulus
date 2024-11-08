<!-- markdownlint-disable MD043 -->
# NVIDIA Modulus Examples

## Introduction

This repository provides sample applications demonstrating use of specific Physics-ML
model architectures that are easy to train and deploy. These examples aim to show how
such models can help solve real world problems.

## Modulus Concepts

|Use case|Model|Transient|
| --- | --- |  --- |
|[Darcy Flow](./cfd/darcy_fno/)|FNO|NO|

## CFD

|Use case|Model|Transient|
| --- | --- |  --- |
|[Vortex Shedding](./cfd/vortex_shedding_mgn/)|MeshGraphNet|YES|
|[Drag prediction - Ahmed body](./cfd/aero_graph_net/)|MeshGraphNet|NO|
|[Flow prediction - DriveML](./cfd/xaeronet/)|MeshGRaphNet, Unet|NO|
|[Navier-Stokes Flow](./cfd/navier_stokes_rnn/)|RNN|YES|
|[Gray-Scott System](./cfd/gray_scott_rnn/)|RNN|YES|
|[Lagrangian Fluid Flow](./cfd/lagrangian_mgn/)|MeshGRaphNet|YES|
|[Darcy Flow using Nested-FNOs](./cfd/darcy_nested_fnos/)|Nested-FNO|NO|
|[Darcy Flow (Data + Physics Driven) using DeepONet approach](./cfd/darcy_physics_informed/)|FNO (branch) and MLP (trunk)|NO|
|[Darcy Flow (Data + Physics Driven) using PINO approach (Numerical gradients)](./cfd/darcy_physics_informed/)|FNO|NO|
|[Stokes Flow (Physics Informed Fine-Tuning)](./cfd/stokes_mgn/)|MeshGraphNet and MLP|NO|

## Weather

|Use case|Model|AMP|CUDA Graphs|Multi-GPU| Multi-Node|
| --- | --- | --- | --- | --- | --- |
|[Medium-range global weather forecast using FCN-SFNO](https://github.com/NVIDIA/modulus-makani)|FCN-SFNO|YES|NO|YES|YES|
|[Medium-range global weather forecast using GraphCast](./weather/graphcast/)|GraphCast|YES|NO|YES|YES|
|[Medium-range global weather forecast using FCN-AFNO](./weather/fcn_afno/)|FCN-AFNO|YES|YES|YES|YES|
|[Medium-range and S2S global weather forecast using DLWP](./weather/dlwp/)|DLWP|YES|YES|YES|YES|
|[Medium-range and S2S global weather forecast using DLWP](./weather/dlwp_healpix/)|DLWP-HEALPix|YES|YES|YES|YES|
|[Medium-range and S2S global weather forecast using DLWP](./weather/dlwp_healpix_coupled/)|DLWP-HEALPix|YES|YES|YES|YES|
|[Medium-range and S2S global weather forecast using DLWP](./weather/pangu_weather/)|Pangu|YES|YES|YES|YES|

## Generative

|Use case|Model|Multi-GPU| Multi-Node|
| --- | --- | --- | --- |
|[Generative Correction Diffusion Model for Km-scale Atmospheric Downscaling](./generative/corrdiff/)|CorrDiff|YES|YES|

## Healthcare

|Use case|Model|Transient|Source|
| --- | --- |  --- | ---|
|[Cardiovascular Simulations](./healthcare/bloodflow_1d_mgn/)|MeshGraphNet|YES|Externally contributed|
|[Brain Anomaly Detection](./healthcare/brain_anomaly_detection/)|FNO|YES| |

## Additive Manufacturing

|Use case|Model|Transient|Source|
| --- | --- |  --- | --- |
|[Metal Sintering Simulation](./additive_manufacturing/sintering_physics/)|MeshGraphNet|YES|Externally contributed|

## Molecular Dymanics

|Use case|Model|Transient|
| --- | --- |  --- |
|[Force Prediciton for Lennard Jones system](./molecular_dynamics/lennard_jones/)|MeshGraphNet|NO|


## Additional examples

In addition to the examples in this repo, more Physics-ML usecases and examples
can be referenced from the [Modulus-Sym examples](https://github.com/NVIDIA/modulus-sym/blob/main/examples/README.md).

## NVIDIA support

In each of the example READMEs, we indicate the level of support that will be provided.
Some examples are under active development/improvement and might involve rapid changes.
For stable examples, please refer the tagged versions.

## Feedback / Contributions

We're posting these examples on GitHub to better support the community, facilitate
feedback, as well as collect and implement contributions using
[GitHub issues](https://github.com/NVIDIA/modulus/issues) and pull requests.
We welcome all contributions!
