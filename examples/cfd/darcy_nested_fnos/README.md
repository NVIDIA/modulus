## Generate Data Sets
Folder `./data` will be created and filled with `out_of_sample.npy`, `training_data.npy`, `validation_data.npy`:
```bash
python generate_nested_darcy.py
```

## Train Models ref0 and ref1
Folders `./checkpoints/ref{X}` will be created in which checkpoints are stored.
Training can be monitored using MLFlow
```bash
python train_nested_darcy.py +model=ref0
python train_nested_darcy.py +model=ref1
```

## Evaluate Model
...using the latest checkpoints.
```bash
python evaluate_nested_darcy.py
```
Note that the combined error norm is lower than the error norm of ref_0 alone.
