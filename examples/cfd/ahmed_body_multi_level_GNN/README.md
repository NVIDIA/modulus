# Multi-Level GNN 

Here we implement the Multi-Level GNN (BSMS-GNN) based on paper [Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network](https://arxiv.org/pdf/2210.02573). It is general for any mesh type, and is fully automatic for creating multi-level meshes.

## Method Details
BSMS-GNN has two major building blocks: 1.) Bi-Stride Pooling and Adjacency Enhancement; 2.) Transition between levels. The first part precomputes different levels of meshes consecutively from the input mesh as the top level, requiring the input of how many levels you want. The second part determines how to do message passing across levels, basically determine the how to compute the edge weight and node updating after pooling and returning.

## Code Structure

The main code entry is `train_multi_level_meshes.py`. It first precomputed all multi-level meshes for input data and then do the training. The configuration file needs to specify how many levels you want, with `mesh_layer`, and where do you want to store those precomputed results, with `multi_mesh_data_dir`.


### Precomputing the multi-level meshes.
This part is handled with `multi_mesh_save_and_load.py`, which mainly depends on `bsms_graph_wrapper.py`. `bsms_graph_wrapper.py` first decide the starting node to compute BFS and every time the graph will be partitioned into two groups, and preserve only one group in the next level. The current starting node is selected by the conter of the input mesh, this can be changed later based on different applications.

### Transition and Message passing across levels.
This part is wrapped into `models.py`. This is a similar class as MeshGraphNet, with the only difference in that there is a `BS_process`. It iteratively (with for loop) computes the output node features for each level, and do downsampling, upsampling. The basic operations for level transitions are defined in `ops.py`

## Current Results
- The original MeshGraphNets: test error is 21.48797219246626%. (19 sec per epoch)
- BSMS-layer=4: test error is ls 16.936930902302265%. (82 sec per epoch)
- BSMS-layer=6: test error is ls 12.11720959842205%. (95 sec per epoch)
## Some problems to be fixed

- The current implementation only works for `batch_size = 1`.
- The orginal meshgraphnet generats denormalized testing error for 21.48797219246626 with 500 epochs. It has a mismatch with the one reported on the current github inference folder.
- When the number of level increases to 8, there seems to be some error. There may need some check towards the multi-level meshes generation process, to see if at some point, the deepest level contains no node, etc.


## Getting Started
To train the model, first install `pip install sparse_dot_mkl`, and run

```bash
python train_multi_level_meshes.py
```

The script I use for running on OCI is 

```bash
NUM_GPUS=8 sbatch --ntasks=8 --gres=gpu:8 --time=04:00:00 BSMS.sh
```
```bash
#!/bin/bash
#SBATCH -A xxx
#SBATCH -J BSMS_6
#SBATCH -t 04:00:00
#SBATCH -p polar
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --dependency=singleton

: "${NUM_GPUS:=1}"
: "${NUM_CPUS_PER_TASK:=16}"

# Code directory (change this to your own code directory)
readonly _code_root="xxx"

# mount the data and code directories
readonly _cont_mounts="${_code_root}:/code:rw"

# Pull image from NGC registry
readonly _count_image='nvcr.io/nvidia/modulus/modulus:24.04'


RUN_CMD="python train_multi_level_meshes.py"

srun    --nodes=1               \
        --ntasks=${NUM_GPUS}    \
        --gpus=${NUM_GPUS}      \
        --gpus-per-node=${NUM_GPUS}             \
        --cpus-per-task=${NUM_CPUS_PER_TASK}    \
        --exclusive                             \
        --container-image=${_count_image}       \
        --container-mounts=${_cont_mounts}      \
        bash -c "
        cd xxx
        pip install sparse_dot_mkl
        ${RUN_CMD}"
```

