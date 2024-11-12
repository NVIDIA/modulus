# US-DLWP analysis 

The code here is aimed and streamlining the analysis of HRRR models. The analysis directory containes file needed to run an analysis and the report directory will serve the report with the run files. 

The main file is `run_analysis.py` which recives spcification from an `--analysis_config`, typically `analysis_config.json`; a `--case_studies_file` typicaly `case_studies.json`; and a `--registry_file` with specs of the analysis including model, number of steps and details of case studies. The `case_studies.json` is hard coded generally should not be changed, the `analysis_config.json` can be copied and changed by the user. 
To run the analysis do:
```
python analysis/run_analysis.py  --analyis_config analysis/analysis_config.json --case_studies_file analysis/case_studies.json --registry_file config/registry.json
```
The call produces csv and netcdf files that are read by the report. To add a case study simply edit `case_studies.json` with date times of initial condition and index of the max/min of lat and lon for the area of interest.

To serve the report: `python report.py` and it will provide the link in terminal.


## Sweeps

Sweeps allow you to sweep over a single parameter of the inference model. To run a sweep, do the following.

Create a sweep file by following the template `analysis/sweep_config.json`. 

```
{
  "registry_config_name": "diffusion_regression_a2a_v3_1_exclude_w",
  "sweep_parameter": "posthoc_ema_sigma",
  "sweep_values": [0.4, 0.2, 0.1, 0.05, 0.01, 0.001],
  "sweep_tags": ["40pc", "20pc", "10pc", "5pc", "1pc", "01pc"]
}
```

`registry_config_name` is the name of the model in the main registry that you want to test. 
`sweep_parameter` is the parameter you want to sweep over. Here I want to sweep over the `posthoc_ema_sigma` parameter.
`sweep_values` are the values you want to sweep over. Here I want to sweep over the values `[0.4, 0.2, 0.1, 0.05, 0.01, 0.001]`.
`sweep_tags` are the tags you want to give to the sweep. Here I want to give the tags `["40pc", "20pc", "10pc", "5pc", "1pc", "01pc"]`.

To test the sweep using a dry run do the following.

```
module load python #loads the nersc python module
python analysis/inference_sweep.py --sweep_config analysis/sweep_config.json --sweep_name ema --dry_run
```

This will generate a directory with the analysis configs for each member of the sweep. It will also genearate a derived model registry which will contain the model details with the sweep parameter looped over. The main registry will not be touched or modified in any way.

To run the sweep do one of the following.

- Use the `submit_sweep_jobs.sh` to submit sweep jobs that you created config files for in the previous step (dry_run). This script will submit the jobs to the queue and run them in parallel. (Recommended)

You can also submit the jobs directly by removing the dryrun flag.

- Check the batch_inference_sweep.sh file to make sure the resources are correct for your job.
- Run the following commands

```
module load python #loads the nersc python module
python analysis/inference_sweep.py --sweep_config analysis/sweep_config.json --sweep_name ema
```

You can also run the sweep in interactive mode by adding the --interactive flag to inference_sweep.py.



