# MeshGraphNet with Lagrangian mesh

This is an example of Meshgraphnet for particle-based simulation on the
water dataset based on
<https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate>
in PyTorch.
It demonstrates how to train a Graph Neural Network (GNN) for evaluation
of the Lagrangian fluid.

## Problem overview

In this project, we provide an example of Lagrangian mesh simulation for fluids. The
Lagrangian mesh is particle-based, where vertices represent fluid particles and
edges represent their interactions. Compared to an Eulerian mesh, where the mesh
grid is fixed, a Lagrangian mesh is more flexible since it does not require
tessellating the domain or aligning with boundaries.

As a result, Lagrangian meshes are well-suited for representing complex geometries
and free-boundary problems, such as water splashes and object collisions. However,
a drawback of Lagrangian simulation is that it typically requires smaller time
steps to maintain physically valid prediction.

## Dataset

We rely on [DeepMind's particle physics datasets](https://sites.google.com/view/learning-to-simulate)
for this example. They datasets are particle-based simulation of fluid splashing
and bouncing in a box or cube.

| Datasets     | Num Particles | Num Time Steps |    dt    | Ground Truth Simulator |
|--------------|---------------|----------------|----------|------------------------|
| Water-3D     | 14k           | 800            | 5ms      | SPH                    |
| Water-2D     | 2k            | 1000           | 2.5ms    | MPM                    |
| WaterRamp    | 2.5k          | 600            | 2.5ms    | MPM                    |

## Model overview and architecture

In this model, we utilize a Meshgraphnet to capture the fluid system’s dynamics.
We represent the system as a graph, with vertices corresponding to fluid particles
and edges representing their interactions. The model is autoregressive, using
historical data to predict future states. The input features for the vertices
include the current position, current velocity, node type (e.g., fluid, sand,
boundary), and historical velocity. The model's output is the acceleration,
defined as the difference between the current and next velocity. Both velocity
and acceleration are derived from the position sequence and normalized to a
standard Gaussian distribution for consistency.

For computational efficiency, we do not explicitly construct wall nodes for
square or cubic domains. Instead, we assign a wall feature to each interior
particle node, representing its distance from the domain boundaries. For a
system dimensionality of \(d = 2\) or \(d = 3\), the features are structured
as follows:

- **Node features**: position (\(d\)), historical velocity (\(t \times d\)),
                     one-hot encoding of node type (6), wall feature (\(2 \times d\))
- **Edge features**: displacement (\(d\)), distance (1)
- **Node target**: acceleration (\(d\))

We construct edges based on a predefined radius, connecting pairs of particle
nodes if their pairwise distance is within this radius. During training, we
shuffle the time sequence and train in batches, with the graph constructed
dynamically within the dataloader. For inference, predictions are rolled out
iteratively, and a new graph is constructed based on previous predictions.
Wall features are computed online during this process. To enhance robustness,
a small amount of noise is added during training.

The model uses a hidden dimensionality of 128 for the encoder, processor, and
decoder. The encoder and decoder each contain two hidden layers, while the
processor consists of eight message-passing layers. We use a batch size of
20 per GPU, and summation aggregation is applied for message passing in the
processor. The learning rate is set to 0.0001 and decays exponentially with
a rate of 0.9999991. These hyperparameters can be configured in the config file.

## Getting Started

This example requires the `tensorflow` library to load the data in the `.tfrecord`
format. Install with

```bash
pip install tensorflow
```

To download the data from DeepMind's repo, run

```bash
cd raw_dataset
bash download_dataset.sh Water /data/
```

Change the data path in `conf/config_2d.yaml` correspondingly

To train the model, run

```bash
python train.py
```

Progress and loss logs can be monitored using Weights & Biases. To activatethat,
set `wandb_mode` to `online` in the `conf/config_2d.yaml` This requires to have an active
Weights & Biases account. You also need to provide your API key in the config file.

```bash
wandb_key: <your_api_key>
```

The URL to the dashboard will be displayed in the terminal after the run is launched.
Alternatively, the logging utility in `train.py` can be switched to MLFlow.

Once the model is trained, run

```bash
python inference.py
```

This will save the predictions for the test dataset in `.gif` format in the `animations`
directory.

## References

- [Learning to simulate complex physicswith graph networks](arxiv.org/abs/2002.09405)
- [Dataset](https://sites.google.com/view/learning-to-simulate)
- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
