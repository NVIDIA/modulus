# GraphCast for weather forecasting

This example is a lightweight implementation of the DeepMind's
[GraphCast](https://arxiv.org/abs/2212.12794) model in PyTorch, to provide a recipe on
how to train such GNN models in Modulus.

## Problem overview

GraphCast is a multi-scale graph neural network-based autoregressive model. It is
trained on historical weather data from ECMWF's ERA5 reanalysis archive. GraphCast
generates predictions at 6-hour time steps for a set of surface and atmospheric
variables. This prediction covers a 0.25-degree latitude-longitude grid,
providing approximately 25 x 25 kilometer resolution at the equator.

## Dataset

The model is trained on a 73-channel subset of the ERA5 reanalysis data on single levels
and pressure levels that are pre-processed and stored into HDF5 files.
A 20-channel subset of the ERA5 training data is hosted at the
National Energy Research Scientific Computing Center (NERSC). For convenience
[it is available to all via Globus](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F).
You will need a Globus account and will need to be logged in to your account in order
to access the data.  You may also need the [Globus Connect](https://www.globus.org/globus-connect)
to transfer data.

## Model overview and architecture

Please refer to the [reference paper](https://arxiv.org/abs/2212.12794) to learn about
the model architecture.

## Getting Started

To train the model on a single GPU, run

```bash
python train_graphcast.py
```

This will launch a GraphCast training with up to 12 steps of fine-tuning using the base
configs specified in the `constants.py` file.

Data parallelism is also supported with multi-GPU runs. To launch a multi-GPU training,
run

```bash
mpirun -np <num_GPUs> python train_graphcast.py
```

If running in a docker container, you may need to include the `--allow-run-as-root` in
the multi-GPU run command.

Progress and loss logs can be monitored using Weights & Biases. This requires to have an
active Weights & Biases account. You also need to provide your API key. There are
multiple ways for providing the API key but you can simply export it as an environment
variable

```bash
export WANDB_API_KEY=<your_api_key>
```

The URL to the dashboard will be displayed in the terminal after the run is launched.

If needed, Weights & Biases can be disabled by

```bash
export WANDB_MODE='disabled'
```

## References

- [GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794)
