from utils.diffusions import networks, losses
import torch


def test_edm_no_conditioning():
    """run all the EDM loss and unet"""
    in_channels = 3
    resolution = 64
    net = networks.get_preconditioned_architecture(
        "ddpmpp-cwb-v0", resolution=resolution, target_channels=in_channels
    )

    x = torch.zeros((1, in_channels, resolution, resolution))
    sigma = torch.ones([1])
    y = net(x, sigma)
    assert y.shape == (1, in_channels, resolution, resolution)


def test_edm_conditioning():
    """run all the EDM loss and unet"""
    resolution = 64
    target_channels = 3
    cond_channels = 2

    net = networks.get_preconditioned_architecture(
        "ddpmpp-cwb-v0",
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=cond_channels,
    )

    x = torch.zeros((1, target_channels, resolution, resolution))
    condition = torch.zeros((1, cond_channels, resolution, resolution))

    sigma = torch.ones([1])
    y = net(x, sigma, condition=condition)
    assert y.shape == (1, target_channels, resolution, resolution)


def test_edm_loss():
    loss = losses.EDMLoss()
    resolution = 64
    target_channels = 3
    cond_channels = 2
    batch = 2

    net = networks.get_preconditioned_architecture(
        "ddpmpp-cwb-v0",
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=cond_channels,
    )

    x = torch.zeros((batch, target_channels, resolution, resolution))
    condition = torch.zeros((batch, cond_channels, resolution, resolution))
    out = loss(net, x, condition=condition)
    assert out.shape == x.shape
