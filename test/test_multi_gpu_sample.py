import torch
import pytest

def test_multi_gpu():
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 1, "Not enough GPUs available for test"

    for i in range(num_gpus):
        with torch.cuda.device(i):
            tensor = torch.tensor([1., 2., 3.], device=f'cuda:{i}')
            assert tensor.sum() == 6.0

if __name__ == "__main__":
    pytest.main([__file__])