# Temporal attention model in Mesh-Reduced space for transient vortex shedding

This example is an implementation of the paper "Predicting Physics in Mesh-reduced Space
with Temporal Attention" in PyTorch.
It demonstrates how to train a Graph Neural Network (GNN) as encoder to compress the high-dimensional
physical state into latent space and apply a multi-head attention model for temporal predictions for 
the transient vortex shedding on parameterized geometries.

## Problem overview


## Dataset

We use vortex shedding dataset for this example. The dataset includes
51 training, and 50 test samples that are simulated using OpenFOAM
with irregular triangle 2D meshes, each for 401 time steps with a time step size of
0.5s. These samples vary in the Reynolds number. Each sample share the same mesh with
1699 nodes.

## Model overview and architecture

The model is auto-regressive. It first encodes the graph state into a latent vector via a Graph
Nueral Network. Then a multi-head temporal model takes the initial condition tokens and pysical paramerters
as the input and predicts the solution for the following sequence in the latent space just like a language model. 

The model uses the input mesh to construct a bi-directional DGL graph for each sample.
The node features include (3 in total):

- Velocity components at time step $t$, i.e., $u_t$, $v_t$
- Pressure at time step $t$, $p_t$

The edge features for each sample are time-independent and include (3 in total):

- Relative $x$ and $y$ distance between the two end nodes of an edge
- L2 norm of the relative distance vector

The output of the model is the velocity components for the following steps, i.e.,
$\{\ldots, (u_{t}$, $v_{t}), (u_{t+1}$, $v_{t+1}), \ldots\}$, as well as the pressure $\{\ldots,p_{t},p_{t+1}\,\ldots}$.


For the PbGMR-GMUS, a hidden dimensionality of 128 is used in the encoder, and decoder. 
The encoder and decoder consist of two hidden layers. Batch size per GPU is set to 1 for the encoding-decoding process.
Mean aggregation is used in the processor for message aggregation. A learning rate of 0.0001 is used, decaying
exponentially with a rate of 0.9999991. Traing epochs is set as 300.

For the multi-head attention temporal model, the dimension for each token is $3 \times 256 = 768$. The hidden dimension usded in
<<<<<<< HEAD
the temporal model is $4 \times 768 = 4072$. The number of head is 8. Batch size per GPU is set to 10 for the sequence model training. Traing epochs is set as 200000.
=======
the temporal model is $4 \times 768 = 4072$.
>>>>>>> 4ce71613df08f3f75321e50e76f2abae5bda3c65

## Getting Started



To download the data , run

```bash
cd dataset
<<<<<<< HEAD
sh TODO: need a shell to download the raw dataset from a cloud storage.
```

To train the encoding-decoding model, run
=======
sh TODO: need a shell to download the raw dataset
```

To train the model, run
>>>>>>> 4ce71613df08f3f75321e50e76f2abae5bda3c65

```bash
python train.py
```

<<<<<<< HEAD
To test the reconstruction error, run 

```bash
python test.py
```

To train the sequence model, run 

```bash
python train_sequence.py
```



=======
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
>>>>>>> 4ce71613df08f3f75321e50e76f2abae5bda3c65

Once the model is trained, run

```bash
<<<<<<< HEAD
python test_sequence.py
```


## References

- [Predicting Physics in Mesh-reduced Space with Temporal Attention](https://arxiv.org/abs/2201.09113)
=======
python inference.py
```

This will save the predictions for the test dataset in `.gif` format in the `animations`
directory.

## References

- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
>>>>>>> 4ce71613df08f3f75321e50e76f2abae5bda3c65
