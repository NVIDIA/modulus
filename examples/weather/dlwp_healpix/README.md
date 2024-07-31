# Deep Learning Weather Prediction (DLWP-HEALPIX) model for weather forecasting

This example is an implementation of the
[DLWP HEALPix](https://arxiv.org/abs/2311.06253)
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
