# Internal storage

This directory stores code, scripts, and other artifacts that are used internally
and in general, should not be released externally. For example, custom Docker images,
ORD cluster scripts and other similar things should be stored here.

1. [Docker](#docker)
    1. [Building the image](#building-the-image)
    2. [Running locally](#running-the-container-locally)
    3. [Running TPUNet training](#running-tpunet-training-script)
2. [Running jobs in ORD](#running-jobs-in-ord-cluster)

## Docker

TriplaneUNet code uses Docker to run the experiments and other tasks
across various environments.
TPUNet Docker is based off of official [Modulus image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus/tags),
with few minor modifications, such as dependencies and configurations.

When the image is built, Modulus source code in the current directory
is included in the image. This allows running the image in any environment,
for example, locally or in ORD/NGC and expect the same results.

### Building the image

**Note**: this Docker images depends on a pre-built Python wheel which is stored in Git LFS.
Make sure that you have Git LFS installed, check by running `git lfs ls-files`.
If not - install and do the `git pull` to get the necessary files.

Use [build_docker.sh](./docker/build_docker.sh) to build and push the image:

```bash
./docker/build_docker.sh
```

The script takes one optional parameter - the full name of the Docker image.
If not provided, a default will be used: `nvcr.io/nv-maglev/dlav/modulus-tpunet-${USER}:latest`

The first time the script may take a few minutes to run due to image pulling
and dependencies installation. Subsequent launches should be fast, given that only the
contents of Modulus directory is changed.

### Running the container locally

To run the TPUNet Docker container locally, use the convenience script,
[run_docker.sh](./docker/run_docker.sh). The script takes one optional
parameter - the command to execute when the container is created.
If no parameter is provided, `bash` will be used.
The script can be parameterized by defining environment variables.

```text
$ ./internal/docker/run_docker.sh

Image name (IMAGE_FULL_NAME)       : nvcr.io/nv-maglev/dlav/modulus-tpunet-akamenev:latest
Container user (CONTAINER_USER)    : du
Container mounts (CONTAINER_MOUNTS): /data/:/data
Container name (CONTAINER_NAME)    : tpunet-transient
...
```

For example, to override container mounts:

```bash
CONTAINER_MOUNTS="/data/src/modulus/:/modulus" ./internal/docker/run_docker.sh
```

multiple mounts can be provided using comma-separated list.

#### Running TPUNet training script

To run the training script, just pass it along with its parameters
to the `run_docker.sh` script:

```bash
./internal/docker/run_docker.sh \
    python train.py \
    +experiment=drivaer/triplane_unet \
    data.data_path=/data/src/modulus/data/triplane_unet/drivaer/ \
    'data.subsets_postfix=[nospoiler]' \
    ~loggers.wandb \
    'model.hidden_channels=[16, 16, 16, 16]' \
    'output=/data/src/modulus/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
```

The paths in `data.data_path` and `output` should be accessible from the Docker container
i.e. properly mounted via `CONTAINER_MOUNTS` variable. `output` parameter uses standard
[Hydra resolver](https://hydra.cc/docs/configure_hydra/intro/#resolvers-provided-by-hydra).

## Running jobs in ORD cluster

[ORD](https://confluence.nvidia.com/display/HWINFCSSUP/CS-OCI-ORD+FAQ)
is an internal GPU [SLURM cluster](https://slurm.schedmd.com/overview.html).

Example of command that launches a single-GPU training job:

```bash
NUM_GPUS=1 sbatch ./ord/train.sbatch
```

**Note**: this command must be launched from one of the ORD login nodes,
see ORD FAQ for more details.

See [train.sbatch](./ord/train.sbatch) for more details.

### Running multi-GPU training job

TPUNet support data parallel training using standard PyTorch DDP [mechanism](https://pytorch.org/docs/2.2/generated/torch.nn.parallel.DistributedDataParallel.html#).

Example of command that launches 8-GPU training job:

```bash
NUM_GPUS=8 sbatch --ntasks=8 --gres=gpu:8 ./tpunet/train.sbatch loggers.wandb.run_name=TriplaneUNet-8GPU
```

Any additional arguments can be appended to the command line above.

**Note**: if Weights & Biases logger is enabled, make sure to export `WANDB_API_KEY` environment
variable *before* running `sbatch` command.

GPU usage can be monitored using [Grafana dashboard](https://grafana.nvidia.com/d/fdimc4dtoyfpcf/draco-oci-clusters-dcgm?orgId=10&var-cluster=cs-oci-ord).
Filter **Node** using the name of the node from `scontrol show job` command.
