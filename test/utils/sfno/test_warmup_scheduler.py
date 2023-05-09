import torch
from torch import nn
from torch.optim import lr_scheduler as lrs

from modulus.utils.sfno.warmup_scheduler import WarmupScheduler

def test_warmup_scheduler():
    """ test warmup scheduler"""

    param = nn.Parameter(torch.zeros((10), dtype=torch.float))
    opt = torch.optim.Adam([param], lr=0.5)

    start_lr = 0.01
    num_warmup = 10
    num_steps = 20

    main_scheduler = lrs.CosineAnnealingLR(opt, num_steps, eta_min=0)
    scheduler = WarmupScheduler(main_scheduler, num_warmup, start_lr)

    for epoch in range(num_steps + num_warmup):
        scheduler.step()

    sd = scheduler.state_dict()
    scheduler.load_state_dict(sd)
    assert torch.allclose(torch.tensor(scheduler.get_last_lr()[0]), torch.tensor(0.0))

