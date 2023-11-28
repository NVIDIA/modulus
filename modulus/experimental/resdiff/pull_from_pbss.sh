#!/bin/bash


items=(
    checkpoints
)

mkdir -p data
for item in ${items[*]}
do
    aws s3 sync s3://cwb-diffusions/$item/ data/$item
done
