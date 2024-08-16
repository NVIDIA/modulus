# Transolver for Darcy Flow

This example demonstrates how to set up a data-driven model for a 2D Darcy flow using
the Transolver inside of Modulus.
Training progress can be tracked through [MLFlow](https://mlflow.org/docs/latest/index.html).
This example runs on a single GPU.

## Getting Started

To train the model, simply run

```bash
python train_transolver_darcy.py
```

To reproduce the results in the paper, run

```bash
python train_transolver_darcy_fix.py
```

## Additional Information

## References

- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)
