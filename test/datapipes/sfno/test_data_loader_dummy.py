import pytest
from unittest.mock import MagicMock

from modulus.datapipes.climate.sfno.dataloaders.data_loader_dummy import DummyLoader

def test_dummyloader():
    # Create a mock object for params
    params = MagicMock()
    params.dt = 1
    params.batch_size = 1
    params.n_history = 1
    params.n_future = 1
    params.in_channels = [1]
    params.out_channels = [1]
    params.roll = True
    params.io_grid = [1, 1, 1]
    params.io_rank = [1, 1, 1]
    params.crop_size_x = 1
    params.crop_anchor_x = 0
    params.img_shape_x = 1
    params.crop_size_y = 1
    params.crop_anchor_y = 0
    params.img_shape_y = 1
    params.n_years = 1
    params.n_samples_per_year = 1
    
    # Define dummy arguments
    location = 'none'
    train = True
    device = 'cuda:0'

    # Create an instance of the class
    dummyloader = DummyLoader(params, location, train, device)

    # Check if the object is initialized correctly
    assert dummyloader.dt == params.dt
    assert dummyloader.batch_size == params.batch_size
    assert dummyloader.n_history == params.n_history
    assert dummyloader.n_future == params.n_future
    assert dummyloader.in_channels == params.in_channels
    assert dummyloader.out_channels == params.out_channels
    assert dummyloader.roll == params.roll
    assert dummyloader.io_grid == params.io_grid
    assert dummyloader.io_rank == params.io_rank
    assert dummyloader.location == location
    assert dummyloader.train == train
    #assert dummyloader.device == device

    # Test the __len__ method
    assert len(dummyloader) == 1

    # Test the __iter__ method
    assert iter(dummyloader) == dummyloader

    # Test the get_input_normalization method
    in_bias, in_scale = dummyloader.get_input_normalization()
    assert in_bias.shape == (1, len(params.in_channels), 1, 1)
    assert in_scale.shape == (1, len(params.in_channels), 1, 1)

    # Test the get_output_normalization method
    out_bias, out_scale = dummyloader.get_output_normalization()
    assert out_bias.shape == (1, dummyloader.n_out_channels_local, 1, 1)
    assert out_scale.shape == (1, dummyloader.n_out_channels_local, 1, 1)

