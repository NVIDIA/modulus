## Generate Data Sets
folder ./data will be created and filled with out_of_sample.npy training_data.npy validation_data.npy:
```bash
python generate_nested_darcy.py
```

## train models ref0 and ref1
folders ./checkpoints/ref{X} will be created in which checkpoints are stored.
Training can be monitored using MLFlow
```bash
python train_nested_darcy.py +model=ref0
python train_nested_darcy.py +model=ref1
```

## evaluate model
...using latest checkpoints
```bash
python evaluate_nested_darcy.py
```
