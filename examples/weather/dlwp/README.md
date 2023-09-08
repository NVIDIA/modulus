# Deep Learning Weather Prediction (DLWP) model for weather forecasting

This example is an implementation of the
[DLWP Cubed-sphere](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002502)
model. The DLWP model can be used to predict the state of the atmosphere given a previous
atmospheric state.  You can infer a 320-member ensemble set of six-week forecasts at 1.4°
resolution within a couple of minutes, demonstrating the potential of AI in developing
near real-time digital twins for weather prediction

## Problem overview

The goal is to train an AI model that can emulate the state of the atmosphere and predict
global weather over a certain time span. The Deep Learning Weather Prediction (DLWP) model
uses deep CNNs for globally gridded weather prediction. DLWP CNNs directly map u(t) to
its future state u(t+Δt) by learning from historical observations of the weather,
with Δt set to 6 hr

## Dataset

The model is trained on 7-channel subset of ERA5 Data that is mapped onto a cubed sphere
grid with a resolution of 64x64 grid cells. The map files were generated using
[TempestRemap](https://github.com/ClimateGlobalChange/tempestremap) library.
The model uses years 1980-2015 for training, 2016-2017 for validation
and 2018 for out of sample testing. Some sample scripts for downloading the data and processing
it are provided in the `data_curation` directory. A larger subset of dataset is hosted
at the National Energy Research Scientific Computing Center (NERSC). For convenience
[it is available to all via Globus](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F).
You will need a Globus account and will need to be logged in to your account in order
to access the data. You may also need the [Globus Connect](https://www.globus.org/globus-connect)
to transfer data.

## Model overview and architecture

DLWP uses convolutional neural networks (CNNs) on a cubed sphere grid to produce global forecasts.
The latest DLWP model leverages a U-Net architecture with skip connections to capture multi-scale
processes.The model architecture is described in the following papers

[Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather Prediction Models](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002502)

[Improving Data-Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002109)

## Getting Started

To train the model, run

```bash
python train_dlwp.py
```

Progress can be monitored using MLFlow. Open a new terminal and navigate to the training
directory, then run:

```bash
mlflow ui -p 2458
```

View progress in a browser at <http://127.0.0.1:2458>

## References

[Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather Prediction Models](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002502)

[Arbitrary-Order Conservative and Consistent Remapping and a Theory of Linear Maps: Part 1](https://journals.ametsoc.org/view/journals/mwre/143/6/mwr-d-14-00343.1.xml)

[Arbitrary-Order Conservative and Consistent Remapping and a Theory of Linear Maps, Part 2](https://journals.ametsoc.org/view/journals/mwre/144/4/mwr-d-15-0301.1.xml)
