# Nested Fourier Neural Operater for Darcy Flow

This example demonstrates how to set up a data-driven model for a 2D Darcy flow using
the Nested Fourier Neural Operator (FNO) architecture inside of Modulus.

## Getting Started

Start with generating the dataset for training:

```bash
python generate_nested_darcy.py
```

which will create the folder `./data` with `out_of_sample.npy`, `training_data.npy`, `validation_data.npy`.

To train the model, run

```bash
python train_nested_darcy.py +model=ref0
python train_nested_darcy.py +model=ref1
```

To evaluate the model use:

```bash
python evaluate_nested_darcy.py
```

## Additional Information

## References

- [Real-time high-resolution CO2 geological storage prediction using nested Fourier neural operators](https://arxiv.org/abs/2210.17051)
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
