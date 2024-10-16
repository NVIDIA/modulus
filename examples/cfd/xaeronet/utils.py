import os
import torch

def find_bin_files(data_path):
    """
    Finds all .bin files in the specified directory.
    """
    return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.bin')]
    
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
