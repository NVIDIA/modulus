<!-- markdownlint-disable -->
## Selene Usage

### Selene basics

To access Selene, you need to first request access here: https://docs.google.com/forms/d/e/1FAIpQLSdtmzM8eHm8cPw-No8PpZseWYI_H-ExKWJ3kAe0i5Nnsp2qmQ/viewform. The correct groups to ask for are `sw_climate_fno` and `sw_earth2_ml`. Once this is done, you can read on how to access Selene here: https://confluence.nvidia.com/pages/viewpage.action?spaceKey=HWINFCSSUP&title=Accessing+Selene+Cluster


### Setting up era5

Once you have access to Selene, download the code to your home directory and 

### Running Jobs on Selene

#### Preallocated jobs

For quick testing, it is useful to pre-allocate a slurm job first and then run commands attaching them to the preallocated job. To do so, we can request a node for an hour in the following way:

```
salloc -p interactive -A sw_earth2_ml -J sw_earth2_ml-sfno:sfno_training -t 01:00:00 -N1 bash
```

This will create a job on the interactive partition for one hour. Notice the account and job name specifications which need to follow the prescribed patter. Once your allocation goes through, slurm will return a jobid:
```
salloc: Pending job allocation 3520410
salloc: job 3520410 queued and waiting for resources
salloc: job 3520410 has been allocated resources
salloc: Granted job allocation 3520410
salloc: Waiting for resource configuration
salloc: Nodes luna-0197 are ready for job
```
We copy-paste this job id into the script `preallocated_srun.sh` and adapt the line
```
export ALLOCATED_JOBID=3520410
```
Now make sure that the variables
```
readonly TRAINING_DATA="/lustre/fsw/sw_climate_fno/34Vars"
readonly USER_OUTPUT="/lustre/fsw/pathtoyouruserdirectory"
readonly STATS="/lustre/fsw/sw_climate_fno/34Vars_statsv2/stats"
readonly INVARIANTS="/lustre/fsw/sw_climate_fno/test_datasets/48var-6hourly/invariants"
readonly CODE="/home/username/era5_wind"
readonly PLOTS="/home/username/plots"
```
are correctly set. The important variables that you most likely need to modify are the paths `TRAINING_DATA`, `USER_OUTPUT` and `CODE`, which point to the training data, your output directory and the code directory.

Finally, we need to specify which configuration to run by adapting the lines
```
readonly _config_file=./config/sfnonet.yaml
readonly _config_name=sfno_baseline_linear_26ch
```

Congratulations! 🥳 You're now ready to run preallocated runs by executing
```
./preallocated_srun.sh
```

#### Scheduling multi-node jobs

Once you are confident your job is doing the right thing, you can schedule a bigger node. To do so edit the sile `submit_8node.sh` as before. Moreover, we need to adapt the lines
```
#!/bin/bash
#SBATCH -A sw_earth2_ml
#SBATCH --job-name sw_earth2_ml-sfno:sfno_training
#SBATCH -t 03:55:00
#SBATCH -p luna
#SBATCH -N 8
#SBATCH -o sfno-%j.out
#SBATCH --dependency=afterany:100000
```
depending on the desired node cound and job duration.