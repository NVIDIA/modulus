import os
import shutil
import random

# Define the directory that contains the original files
data_dir = 'results'

# Define the directories for your train, validation, and test datasets
train_dir = 'train'
valid_dir = 'validation'
test_dir = 'test'

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all the files in the original directory
all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# Shuffle the files
random.shuffle(all_files)

# Get the count of all files
all_files_count = len(all_files)

# Calculate the size of each dataset
train_size = int(all_files_count * 0.8)
valid_size = int(all_files_count * 0.1)
test_size = all_files_count - train_size - valid_size  # Ensure all files are used

# Split the files
train_files = all_files[:train_size]
valid_files = all_files[train_size:train_size+valid_size]
test_files = all_files[train_size+valid_size:]

# Function to copy files
def copy_files(files, dest_dir):
    for f in files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(dest_dir, f))

# Copy the files
copy_files(train_files, train_dir)
copy_files(valid_files, valid_dir)
copy_files(test_files, test_dir)
