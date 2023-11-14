<!-- markdownlint-disable -->
## Networks

This folder contains models for training and associated code. Models that are currently supported can be queried by calling `networks.models.list_models()`.

### Directory structure
This folder is organized as follows:

```
sfno
├── ...
├── networks                    # code for ML models
│   ├── activations.py          # complex activation functions
│   ├── afnonet_v2.py           # optimized AFNO
│   ├── afnonet.py              # AFNO implementation
│   ├── contractions.py         # einsum wrappers for complex contractions
│   ├── debug.py                # dummy network for debugging purposes
│   ├── layers.py               # MLPs and wrappers for FFTs
│   ├── model_package.py        # model package implementation
│   ├── models.py               # get_model routine and model wrappers for multistep training
│   ├── preprocessor.py         # implementation of preprocessor for dealing with unpredicted channels
│   ├── spectral_convolution.py # spectral convolution layers for (S)FNO architectures
│   ├── sfnonet.py              # implementation of (S)FNO
│   └── Readme.md               # this file
...

```

### Model packages

Model packages are used for seamless inference outside of this repository. They define a flexible interfact which takes care of normalization, unpredicted channels etc. Model packages seemlessly integrate with [earth2mip](https://github.com/NVIDIA/earth2mip).

