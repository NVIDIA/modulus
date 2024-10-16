import os
import torch

def find_bin_files(data_path):
    """
    Finds all .bin files in the specified directory.
    """
    return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.bin')]

def find_h5_files(directory):
    """
    Recursively finds all .h5 files in the given directory.
    """
    h5_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files
    
def save_checkpoint(model, optimizer, scaler, scheduler, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, scaler, scheduler, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: {filename}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {filename}")
        return 0, None

def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): Model to count parameters of.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_continuity_loss(u, sdf):
    """Calculate the continuity residual of a velocity field on a uniform grid."""
    i, j, k = u.shape[2], u.shape[3], u.shape[4]

    # First-order central difference approximations (up to a constant) 
    u__x =  u[:, 0, 2:i, 1:-1, 1:-1] - u[:, 0, 0:i-2, 1:-1, 1:-1]
    v__y =  u[:, 1, 1:-1, 2:j, 1:-1] - u[:, 1, 1:-1, 0:j-2, 1:-1]
    w__z =  u[:, 2, 1:-1, 1:-1, 2:k] - u[:, 2, 1:-1, 1:-1, 0:k-2]

    sdf = sdf[:, 1:-1, 1:-1, 1:-1]
    mask = (sdf > 0).squeeze()

    residual = u__x + v__y + w__z

    return torch.mean(residual[:, mask] ** 2)
