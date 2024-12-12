<!-- markdownlint-disable -->
# NVIDIA Modulus Examples

## Introduction

This repository provides sample applications demonstrating use of specific Physics-ML
model architectures that are easy to train and deploy. These examples aim to show how
such models can help solve real world problems.

## Introductory examples for learning key ideas

|Use case|Concepts covered|
| --- | --- |
|[Darcy Flow](./cfd/darcy_fno/)|Introductory example for learning basics of data-driven models on Physics-ML datasets|
|[Darcy Flow (Data + Physics)](./cfd/darcy_physics_informed/)|Data-driven training with physics-based constraints|
|[Lid Driven Cavity Flow](./cfd/ldc_pinns/)|Purely physics-driven (no external simulation/experimental data) training|
|[Vortex Shedding](./cfd/vortex_shedding_mgn/)|Introductory example for learning the basics of MeshGraphNets in Modulus|
|[Medium-range global weather forecast using FCN-AFNO](./weather/fcn_afno/)|Introductory example on training data-driven models for global weather forecasting (auto-regressive model)|
|[Lagrangian Fluid Flow](./cfd/lagrangian_mgn/)|Introductory example for data-driven training on Lagrangian meshes|
|[Stokes Flow (Physics Informed Fine-Tuning)](./cfd/stokes_mgn/)|Data-driven training followed by physics-based fine-tuning|

## Domain-specific examples

The several examples inside Modulus can be classified based on their domains as below:

> **NOTE:**  The below classification is not exhaustive by any means!
    One can classify single example into multiple domains and we encourage
    the users to review the entire list.

> **NOTE:**  * Indicates externally contributed examples.

### CFD

|Use case|Model|Transient|
| --- | --- |  --- |
|[Vortex Shedding](./cfd/vortex_shedding_mgn/)|MeshGraphNet|YES|
|[Drag prediction - External Aero](./cfd/external_aerodynamics/)|MeshGraphNet, UNet, DoMINO, FigConvNet|NO|
|[Navier-Stokes Flow](./cfd/navier_stokes_rnn/)|RNN|YES|
|[Gray-Scott System](./cfd/gray_scott_rnn/)|RNN|YES|
|[Lagrangian Fluid Flow](./cfd/lagrangian_mgn/)|MeshGraphNet|YES|
|[Darcy Flow using Nested-FNOs](./cfd/darcy_nested_fnos/)|Nested-FNO|NO|
|[Darcy Flow using Transolver*](./cfd/darcy_transolver/)|Transolver (Transformer-based)|NO|
|[Darcy Flow (Data + Physics Driven) using DeepONet approach](./cfd/darcy_physics_informed/)|FNO (branch) and MLP (trunk)|NO|
|[Darcy Flow (Data + Physics Driven) using PINO approach (Numerical gradients)](./cfd/darcy_physics_informed/)|FNO|NO|
|[Stokes Flow (Physics Informed Fine-Tuning)](./cfd/stokes_mgn/)|MeshGraphNet and MLP|NO|
|[Lid Driven Cavity Flow](./cfd/ldc_pinns/)|MLP|NO
|[Magnetohydrodynamics using PINO (Data + Physics Driven)*](./cfd/mhd_pino/)|FNO|YES|
|[Shallow Water Equations using PINO (Data + Physics Driven)*](./cfd/swe_nonlinear_pino/)|FNO|YES|
|[Shallow Water Equations using Distributed GNNs](./cfd/swe_distributed_gnn/)|GraphCast|YES|
|[Vortex Shedding with Temporal Attention](./cfd/vortex_shedding_mesh_reduced/)|MeshGraphNet|YES|

### Weather

|Use case|Model|
| --- | --- |
|[Medium-range global weather forecast using FCN-SFNO](https://github.com/NVIDIA/modulus-makani)|FCN-SFNO|
|[Medium-range global weather forecast using GraphCast](./weather/graphcast/)|GraphCast|
|[Medium-range global weather forecast using FCN-AFNO](./weather/fcn_afno/)|FCN-AFNO|
|[Medium-range and S2S global weather forecast using DLWP](./weather/dlwp/)|DLWP|
|[Medium-range and S2S global weather forecast using DLWP-HEALPix](./weather/dlwp_healpix/)|DLWP-HEALPix|
|[Coupled Ocean-Atmosphere Medium-range and S2S global weather forecast using DLWP-HEALPix](./weather/dlwp_healpix_coupled/)|DLWP-HEALPix|
|[Medium-range and S2S global weather forecast using Pangu](./weather/pangu_weather/)|Pangu|
|[Diagonistic (Precipitation) model using AFNO](./weather/diagnostic/)|AFNO|
|[Unified Recipe for training several Global Weather Forecasting models](./weather/unified_recipe/)|AFNO, FCN-SFNO, GraphCast|
|[Generative Correction Diffusion Model for Km-scale Atmospheric Downscaling](./generative/corrdiff/)|CorrDiff|
|[StormCast: Generative Diffusion Model for Km-scale, Convection allowing Model Emulation](./generative/stormcast/)|CorrDiff|

### Generative

|Use case|Model|
| --- | --- |
|[Fluid Super-resolution*](./generative/diffusion/)|Diffusion|

### Healthcare

|Use case|Model|
| --- | --- |
|[Cardiovascular Simulations*](./healthcare/bloodflow_1d_mgn/)|MeshGraphNet|
|[Brain Anomaly Detection](./healthcare/brain_anomaly_detection/)|FNO|

### Additive Manufacturing

|Use case|Model|
| --- | --- |
|[Metal Sintering Simulation*](./additive_manufacturing/sintering_physics/)|MeshGraphNet|

### Molecular Dymanics

|Use case|Model|
| --- | --- |
|[Force Prediciton for Lennard Jones system](./molecular_dynamics/lennard_jones/)|MeshGraphNet|


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
