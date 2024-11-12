#!/bin/bash

# Use to submit a set of different configs with a series of individual jobs for each

# 80GB A100 
for cfg in swin_73var_p4_wr80_e768_d24_dpr01_lr1em3_abspos_roll_ft swin_73var_p4_wr80_e768_d24_dpr01_lr1em3_roll_ft
do
   name=${cfg}
   for i in {1..1}; do sbatch --job-name=$name --dependency=singleton -C gpu\&hbm80g submit_batch.sh $cfg; done
   echo Submitted $name
done

