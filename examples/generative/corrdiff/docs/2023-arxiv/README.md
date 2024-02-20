<!-- markdownlint-disable -->
Instructions to reproduce the figures for the paper. (all commands run from the project root)

## Contents

```
README.md                              # this file
figures                                # scripts for the figures and tables of the paper
generation_scripts                     # scripts/configs for generating samples
train_and_score_baseline_models.py     # master script for training and scoring the baselines in the table
```

## To reproduce the paper

### Compute the data needed for the plots/tables (a few hours)

To run the big validation used for scoring (this takes a few hours depending on the queues.)

    sbatch generation_scripts/noah/submit_big_validation.sh

Approximately 1/4 of the jobs will fail with a segemntation fault, these need to
be manually removed. To figure out which runs failed, either look at the slurm
logs (`tail -f slurm-<jobid>*.out`), or run `ncdump -h` on the files in
/lustre/fsw/nvresearch/nbrenowitz/diffusions/generations/era5-cwb-v3/validation_big/ranks/.
The broken files have no `prediction` group.  The files are named
{SLURM_ARRAY_TASK_ID}.{RANK}.nc if SLURM_ARRAY_TASK_ID=2 failed, then `rm
2.*.nc` from this folder.

Then concatenate the results into a single zarr (this takes about 5-10 minutes).

    python3 concat.py /lustre/fsw/nvresearch/nbrenowitz/diffusions/generations/era5-cwb-v3/validation_big/ranks/*.nc  /lustre/fsw/nvresearch/nbrenowitz/diffusions/generations/era5-cwb-v3/validation_big/samples.zarr


Finally, you can generate spatial maps of CRPS, RMSE, etc (1 minute)

    python3 score_samples.py /lustre/fsw/nvresearch/nbrenowitz/diffusions/generations/era5-cwb-v3/validation_big/samples.zarr /lustre/fsw/nvresearch/nbrenowitz/diffusions/generations/era5-cwb-v3/validation_big/scores.nc


To compute the baselines run::

    python3 docs/2023-arxiv/train_and_score_baseline_models.py


Finally, get the score table in latex. run the [big-score.ipynb](big-score.ipynb) notebook.

### Create the figures and tabels of the paper O(30 min) to run all

See README in [figures/](./figures/README.md).

### Benchmarking Results (5 minutes)

See [figures/benchmark.py](./figures/benchmark.py)
