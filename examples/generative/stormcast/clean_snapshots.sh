#!/bin/bash

valid_kimg=5500 # this is the last snapshot you want to keep beyond which the loss spiked
snapshot_dir="/pscratch/sd/j/jpathak/hrrr_experiments_eos/0-diffusion_regression_a2a_v3_1_exclude_w-hrrr-gpus64"
batch_factor=64
valid_ema_snapshot=$(($valid_kimg * 1000 / $batch_factor))
echo "valid_ema_snapshot: $valid_ema_snapshot"

#get list of all snapshots with prefix ema-snapshot-*.pkl
snapshots=($(ls $snapshot_dir/ema-snapshot-*.pkl))

for snapshot in ${snapshots[@]}; do
    #extract the file name from the snapshot path
    snapshot_file=$(basename $snapshot)
    #extract the snapshot number from each snapshot file, e.g., ema-snapshot-00066924.pkl
    snapshot_num=$(echo $snapshot_file | grep -o -E '[0-9]+')
    if [ $snapshot_num -gt $valid_ema_snapshot ]; then
        echo "Deleting snapshot: $snapshot"
        rm $snapshot
    fi
done

training_states=($(ls $snapshot_dir/training-state-*.pt))

for state in ${training_states[@]}; do
    state_file=$(basename $state)
    state_num=$(echo $state_file | grep -o -E '[0-9]+')
    if [ $state_num -gt $valid_kimg ]; then
        echo "Deleting training state: $state"
        rm $state
    fi
done

network_snapshots=($(ls $snapshot_dir/network-snapshot-*.pkl))

for network_snapshot in ${network_snapshots[@]}; do
    network_snapshot_file=$(basename $network_snapshot)
    network_snapshot_num=$(echo $network_snapshot_file | grep -o -E '[0-9]+')
    if [ $network_snapshot_num -gt $valid_kimg ]; then
        echo "Deleting network snapshot: $network_snapshot"
        rm $network_snapshot
    fi
done