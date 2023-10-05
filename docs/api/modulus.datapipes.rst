Modulus Datapipes
=================

.. automodule:: modulus.datapipes
.. currentmodule:: modulus.datapipes

Basics
-------

Utilites in this section offers datasets and dataloaders designed to streamline and enhance
data processing operations for a few commonly encountered physical system modeling problems.
The utilites in under ``modulus.datapipes`` can be broadly classified into two types:

1. Datapipes/Dataloaders: These yeild an iterator. 

2. Datasets: These yeild a dataset that can be passed to a dataloader.

.. warning::

    The utilites in this section are under construction, so there might be breaking changes.
    As the new use cases emerge, we are working towards simplifying the API and making
    the utilites more modular. Interested users are also recommended to check out the 
    datapipes in ``modulus.experimental.datapipes``.

Using Climate Datapipes
^^^^^^^^^^^^^^^^^^^^^^^^

Let's see how to use a climate datapipe that is useful for training weather and climate
models. The below code snippet will generate and load a dummy weather data. For resources
on downloading the ERA5 data from the CDS, please refer 
`ERA5 dataset downloader <https://github.com/NVIDIA/modulus-launch/tree/main/examples/weather/dataset_download>`_

The ``ERA5HDF5Datapipe`` datapipe is an optimized data loading pipeline built to specifically
load the ERA5 dataset. This pipeline is built using the GPU-accelerated 
`NVIDIA DALI <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html>`_
library. The datapipe can be configured to yeild input and output samples from the dataset.
It can also automatically normalize the data if the mean and standard-deviation arrays
are provided. Refer to the API docs for 
`modulus.datapipes.climate.era5_hdf5.ERA5HDF5Datapipe <#modulus.datapipes.climate.era5_hdf5.ERA5HDF5Datapipe>`_
for more configuration details. This dataloader loads data asynchronously overlapping the
time required to fetch the data with the other operations like model's forward and backward
pass. 

.. code:: python

    import torch
    import numpy as np
    import h5py

    from modulus.datapipes.climate.era5_hdf5 import ERA5HDF5Datapipe

    def main():
        random_data = np.random.rand(100, 20, 32, 64).astype(
            np.float32
        )  # data in the [N, C, H, W] format

        with h5py.File("./data/2018.h5", "w") as f:
            # create a dataset named "fields" and store the random data
            f.create_dataset("fields", data=random_data)

        dataloader = ERA5HDF5Datapipe(
            data_dir="./data/",
            batch_size=8,
            num_steps=2,
        )

        for data in dataloader:
            print(data[0]["invar"].shape, data[0]["outvar"].shape)
            break

    if __name__ == "__main__":
        main()


Using Benchmark Datapipes
^^^^^^^^^^^^^^^^^^^^^^^^^^

Modulus also provides are few benchmark datapipes for on-the-fly generation and loading
of data. These generate the data on-the-fly using `NVIDIA Warp <https://developer.nvidia.com/warp-python>`_.
These datapipes can be very handy for quick prototyping of models where one wants to try
out different model ideas without worrying about the dataset generation/curation.

Currently support for a Darcy flow datapipe and Kelvin Helmholtz datapipe is available.
These datapipes solve the required equations on the fly to produce the input/output pairs.

Below one such benchmark datapipe.

.. code:: python
 
    from modulus.datapipes.benchmarks.darcy import Darcy2D
    import matplotlib.pyplot as plt

    def main():
        dataloader = Darcy2D(resolution=32, batch_size=2)

        for data in dataloader:
            print(data["permeability"].shape, data["darcy"].shape)
            plt.subplot(1, 2, 1)
            plt.imshow(data["permeability"][0, 0].cpu().numpy(), cmap='gray')   # plot the first sample
            plt.title('Permeability/K')

            plt.subplot(1, 2, 2)
            plt.imshow(data["darcy"][0, 0].cpu().numpy(), cmap='gray')  # plot the first sample
            plt.title('Darcy/Pressure')
            plt.savefig('sample.png')
            break

    if __name__ == "__main__":
        main()

Using Graph Datasets
^^^^^^^^^^^^^^^^^^^^^^

Modulus provides several datasets for using the GNNs in Modulus. Currently datasets for
Ahmed body, vortex shedding and stokes flow are available. These datasets need to be used along with the 
`dgl.dataloading.GraphDataLoader <https://docs.dgl.ai/en/0.8.x/generated/dgl.dataloading.GraphDataLoader.html>`_
to load the data.

<TODO add more details>

.. autosummary::
   :toctree: generated

Benchmark datapipes
-------------------

.. automodule:: modulus.datapipes.benchmarks.darcy
    :members:
    :show-inheritance:

.. automodule:: modulus.datapipes.benchmarks.kelvin_helmholtz
    :members:
    :show-inheritance:

Weather and climate datapipes
-----------------------------

.. automodule:: modulus.datapipes.climate.era5_hdf5
    :members:
    :show-inheritance:

Graph datasets
---------------

.. automodule:: modulus.datapipes.gnn.vortex_shedding_dataset
    :members:
    :show-inheritance:

.. automodule:: modulus.datapipes.gnn.ahmed_body_dataset
    :members:
    :show-inheritance:

.. automodule:: modulus.datapipes.gnn.stokes_dataset
    :members:
    :show-inheritance:

.. automodule:: modulus.datapipes.gnn.utils
    :members:
    :show-inheritance:
