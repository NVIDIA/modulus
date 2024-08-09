# Deep Learning Weather Prediction (DLWP) model for weather forecasting

This example is an implementation of the coupled Ocean-Atmosphere DLWP model.

## Problem overview

The goal is to train an AI model that can emulate the state of the atmosphere and predict
global weather over a certain time span. The Deep Learning Weather Prediction (DLWP) model
uses deep CNNs for globally gridded weather prediction. DLWP CNNs directly map u(t) to
its future state u(t+Δt) by learning from historical observations of the weather,
with Δt set to 6 hr. The Deep Learning Ocean Model (DLOM) that is designed to couple with
deep learning weather prediction (DLWP) model. The DLOM forecasts sea surface
temperature (SST). DLOMs use deep learning techniques as in DLWP models but are
configured with different architectures and slower time stepping. DLOMs and DLWP models
are trained to learn atmosphere-ocean coupling.

## Getting Started

To train the coupled DLWP model, run

```bash
python train.py --config-name config_hpx32_coupled_dlwp
```

To train the coupled DLOM model, run

```bash
python train.py --config-name config_hpx32_coupled_dlom
```
