import matplotlib.pyplot as plt

import shutil
import os
import torch
import params

from concurrent.futures import ProcessPoolExecutor

from PIL import Image


def show_random_datum(train_loader):
    # Get a random batch
    data_iterator = iter(train_loader)
    images, labels = next(data_iterator)

    # Choose a random index within the batch
    random_index = torch.randint(0, len(images), (1,)).item()

    # Extract the random image
    random_image = images[random_index].permute(1, 2, 0).numpy()

    # Display the random image
    plt.imshow(random_image, cmap='gray')
    plt.title(f"Random Image from dataset (Index: {random_index})")
    plt.axis('off')
    plt.show()


def initialise_experiment(args):
    folders = ['samples', 'checkpoints', 'loss_curves']
    if args.sample:
        folders.append(f'runs/{args.model}')
    experiment_path = to_path('experiments', args.experiment_name)

    # Create experiment folders if they don't exist
    for f in folders:
        os.makedirs(to_path(experiment_path, f), exist_ok=True)

    # Make a copy of this run's hyperparameters
    config_name = 'config_sample.json' if args.sample else 'config_train.json'
    config_json_path = to_path(experiment_path, config_name)
    shutil.copy(params.CONFIG_PATH, config_json_path)

    return experiment_path


def crop_image(input_path, output_folder, crop_left_width=150):
    # Get and open image
    filename = os.path.basename(input_path)
    img = Image.open(input_path)

    # Set up the crop. Top left corner is 0,0 and the y-axis is inverted
    width, height = img.size
    left = crop_left_width
    top = 0
    right = width
    bottom = height

    # Execute and save
    cropped_img = img.crop((left, top, right, bottom))

    output_path = to_path(output_folder, f'cropped_{filename}')
    cropped_img.save(output_path)
    print(f'Cropped image: {filename}')


def crop_images_parallel(input_folder, output_folder, crop_left_width=150, num_workers=8):
    os.makedirs(output_folder, exist_ok=True)

    input_paths = [to_path(input_folder, filename) for filename in os.listdir(input_folder)
                   if filename.endswith((".jpg", ".jpeg", ".png"))]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for input_path in input_paths:
            future = executor.submit(crop_image, input_path, output_folder, crop_left_width)
            futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()


def resize_images(input_path, output_path, width=850, height=600):
    # Create the output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # List all files in the input folder
    file_list = os.listdir(input_path)

    for filename in file_list:
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Construct full paths for the input and output images
            input_image_path = to_path(input_path, filename)
            output_image_path = to_path(output_path, filename)

            # Open the original image
            original_image = Image.open(input_image_path)

            # Resize the image
            resized_image = original_image.resize((width, height))

            # Save the resized image to the output folder
            resized_image.save(output_image_path)


def plot_loss_curve(data, epoch, path):
    plt.figure()
    plt.plot(data, label=f'Epoch {epoch}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss against local step in epoch {epoch}')
    plt.savefig(path)


def plot_loss_curves(data, path):
    plt.figure()
    for i, curve in enumerate(data):
        plt.plot(curve, label=f'Epoch {i}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss against local step in each epoch')
    plt.legend()
    plt.savefig(path)


def to_path(*args):
    return os.path.join(*args)
