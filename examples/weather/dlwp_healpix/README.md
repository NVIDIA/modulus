# Deep Learning Weather Prediction (DLWP-HEALPIX) model for weather forecasting

This example is an implementation of the
[DLWP HEALPix](https://arxiv.org/abs/2311.06253)
model. The DLWP model can be used to predict the state of the atmosphere given a previous
atmospheric state.  You can infer a 320-member ensemble set of six-week forecasts at 1.4°
resolution within a couple of minutes, demonstrating the potential of AI in developing
near real-time digital twins for weather prediction. This example also contains an
implementation of the coupled Ocean-Atmosphere DLWP model.

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

To train the DLWP HEALPix model, run

```bash
python train.py
```

To train the coupled DLWP model, run

```bash
python train.py --config-name config_hpx32_coupled_dlwp
```

To train the coupled DLOM model, run

```bash
python train.py --config-name config_hpx32_coupled_dlom
```
