import os
import torch

def main():
    # Check CUDA visibility
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # Ensure you're using the right GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simulate loading model and data
    try:
        # Your model and data loading code here...
        print("Loading model and data...")

        # Simulate model run
        print("Running model...")
        # Your model execution code here...

        print("Model run successful.")

    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()