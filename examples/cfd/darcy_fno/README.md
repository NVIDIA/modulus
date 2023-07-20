# Fourier Neural Operater for Darcy Flow

This example demonstrates how to set up a data-driven model for a 2D Darcy flow using
the Fourier Neural Operator (FNO) architecture inside of Modulus.

## Getting Started

To train the model, run

```bash
python train_fno_darcy.py
```

training data will be generated on the fly.

Progress can be monitored using MLFlow. Open a new terminal and navigate to the training
directory, then run:

```bash
mlflow ui -p 2458
```

View progress in a browser at <http://127.0.0.1:2458>

## Additional Information

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
