import torch
from utils.img_utils import image_to_crops
import pytest


def test_image_to_crops():
    # create a test image tensor
    image = torch.randn(2, 3, 10, 10)

    # test with nx=5, ny=5
    crops = image_to_crops(image, 5, 5)
    assert crops.shape == (8, 3, 5, 5)

    # test with nx=3, ny=5
    # 3 crops in x direction, 2 crops in y direction
    crops = image_to_crops(image, 3, 4)
    assert crops.shape == (12, 3, 3, 4)

    # test with nx=10, ny=10
    crops = image_to_crops(image, 10, 10)
    assert crops.shape == (2, 3, 10, 10)

    # test with nx=20, ny=20
    with pytest.raises(ValueError):
        image_to_crops(image, 20, 20)
