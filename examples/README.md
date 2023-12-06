<!-- markdownlint-disable MD043 -->
# NVIDIA Modulus Examples

## Introduction

This repository provides sample applications demonstrating use of specific Physics-ML
model architectures that are easy to train and deploy. These examples aim to show how
such models can help solve real world problems.

## CFD

|Use case|Model|Transient|Parameterized|
| --- | --- |  --- | --- |
|Vortex Shedding|MeshGraphNet|YES|YES|
|Navier-Stokes Flow|RNN|YES|NO|
|Gray-Scott System|RNN|YES|NO|
|Darcy Flow|FNO|NO|YES|
|Darcy Flow|Nested-FNO|NO|YES|

## Weather

|Use case|Model|AMP|CUDA Graphs|Multi-GPU| Multi-Node|
| --- | --- | --- | --- | --- | --- |
|Medium-range global weather forecast|FCN-SFNO|YES|NO|YES|YES|
|Medium-range global weather forecast|GraphCast|YES|NO|YES|YES|
|Medium-range global weather forecast|FCN-AFNO|YES|YES|YES|YES|
|Medium-range and S2S global weather forecast|DLWP|YES|YES|YES|YES|

## NVIDIA support

In each of the example READMEs, we indicate the level of support that will be provided.
Some examples are under active development/improvement and might involve rapid changes.
For stable examples, please refer the tagged versions.

## Feedback / Contributions

We're posting these examples on GitHub to better support the community, facilitate
feedback, as well as collect and implement contributions using
[GitHub issues](https://github.com/NVIDIA/modulus/issues) and pull requests.
We welcome all contributions!
