export DATA_ROOT=$(pwd)
export MASTER_PORT=29501

# generate coverage statistics so I know what I can delete
python train_diffusions.py --outdir rundir --tick 1

