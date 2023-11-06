#!/usr/bin/env bash
ERA_OUTPUT="/root/data/diffusions/2023-05-31-very-out-of-sample.nc"

WORKSPACE=/workspace
export PYTHONPATH=/root/fcn-mip:$PYTHONPATH

downloadCWB () {
    aws s3 sync s3://cwb-diffusions/checkpoints $WORKSPACE/checkpoints/
}

saveERA5Inputs () {
    [[ -f $ERA_OUTPUT ]] || python3 save_cwb.py $ERA_OUTPUT
}

loadEnv () {
    set -o allexport; source ~/fcn-mip/.env; set +o allexport
}

loadEnv
saveERA5Inputs &
downloadCWB &
wait

(
    cd ../../../ 
    ./test.sh
)
