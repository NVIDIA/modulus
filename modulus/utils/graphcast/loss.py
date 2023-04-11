import torch
import torch.nn as nn


class DefaultLoss(nn.Module):
    """Default loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape (N,).
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        loss = (invar - outvar) ** 2
        loss = loss.mean(dim=-1)
        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss


class CustomLossFunction(torch.autograd.Function):
    """Custom loss function with cell area weighting.

    Parameters
    ----------
    invar : torch.Tensor
        Invar.
    outvar : torch.Tensor
        Outvar.
    area : torch.Tensor
        Cell area with shape (N,).
    """

    @staticmethod
    def forward(ctx, invar, outvar, area):
        with torch.no_grad():
            diff = invar - outvar
            loss = diff**2
            loss = loss.mean(dim=-1)
            loss = torch.mul(loss, area)
            loss = loss.mean()

            loss_grad = 2 * (diff)
            loss_grad *= 1.0 / (invar.size(0) * invar.size(1))
            loss_grad *= area.unsqueeze(-1)
        ctx.save_for_backward(loss_grad)
        return loss

    @staticmethod
    def backward(ctx, _):
        # "grad_output" should be 1, here
        # hence simply ignore
        (grad_invar,) = ctx.saved_tensors
        return grad_invar, None, None


class CustomLoss(nn.Module):
    """Custom loss with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape (N,).
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        loss = CustomLossFunction.apply(invar, outvar, self.area)
        return loss


if __name__ == "__main__":
    dtypes = (torch.float, torch.bfloat16)
    n = (100, 1000)
    d = (100, 200, 400)

    for dt in dtypes:
        for nn in n:
            area = torch.rand(nn, dtype=dt, device="cuda") + 0.01
            default_loss = DefaultLoss(area)
            custom_loss = CustomLoss(area)

            for dd in d:
                invar1 = torch.rand(nn, dd, dtype=dt, device="cuda")
                outvar1 = torch.rand(nn, dd, dtype=dt, device="cuda")

                invar2 = invar1.clone().detach()
                outvar2 = outvar1.clone().detach()

                invar1.requires_grad_()
                invar2.requires_grad_()

                loss1 = default_loss(invar1, outvar1)
                loss1.backward()
                grad1 = invar1.grad

                loss2 = custom_loss(invar2, outvar2)
                loss2.backward()
                grad2 = invar2.grad

                atol = 1.0e-8 if dt == torch.float else 1.0e-6
                loss_diff = torch.abs(loss1 - loss2)
                loss_diff_msg = f"{dt}-{nn}-{dd}: loss diff - min/max/mean: {loss_diff.min()} / {loss_diff.max()} / {loss_diff.mean()}"
                grad_diff = torch.abs(grad1 - grad2)
                grad_diff_msg = f"{dt}-{nn}-{dd}: grad diff - min/max/mean: {grad_diff.min()} / {grad_diff.max()} / {grad_diff.mean()}"

                assert torch.allclose(loss1, loss2, atol=atol), loss_diff_msg
                assert torch.allclose(grad1, grad2, atol=atol), grad_diff_msg
