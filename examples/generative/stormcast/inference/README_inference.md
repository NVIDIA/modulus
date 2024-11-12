## Inference scripts
To run inference from base dir:
* single member inference:
  * inference.py for backbone variables: (see help for other options)
    ```bash
    python inference/inference.py --config afno_backbone --run_num 0
    ```
  * inference_precip.py for total precipitation: (see help for other options)
    ```bash
    python inference/inference.py --config precip --run_num 0
    ```
* ensemble inference:
  * inference_ensemble.py for wind variables: (see help for other options), use submit_batch_ensemble.sh to submit parallelized ensembles across
  different initial conditions
  * inference_ensemble_precip.py for total precipitation: (see help for other options), use submit_batch_ensemble.sh to submit parallelized ensembles across
  different initial conditions; change to inference_ensemble_precip.py in the launch cmd 
