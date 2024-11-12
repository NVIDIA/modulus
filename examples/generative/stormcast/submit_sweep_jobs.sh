#!/bin/bash

sweep_name="regression_kimg"
analysis_configs_dir="analysis/analysis_configs_$sweep_name"

for analysis_config in $(ls $analysis_configs_dir); do
    cmd="python analysis/run_analysis.py --analysis_config $analysis_configs_dir/$analysis_config --case_studies_file analysis/case_studies.json --registry_file analysis/sweep_${sweep_name}_registry.json"
    echo $cmd
    sbatch batch_inference_sweep.sh "$cmd"
done

