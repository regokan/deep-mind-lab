import numpy as np
import torch


def preprocess(image, bkg_color=np.array([144, 72, 17])):
    """
    Preprocess a single image by cropping, background subtraction, and grayscale conversion.

    Args:
        image (ndarray): Input image array with shape (H, W, C).
        bkg_color (ndarray): Background color to subtract, default is [144, 72, 17].

    Returns:
        Tensor: Preprocessed grayscale image as a tensor.
    """
    cropped_image = image[:, :, 10:-10]  # Crop to center region
    grayscale_image = (
        np.mean(cropped_image - bkg_color.reshape(3, 1, 1), axis=0) / 255.0
    )
    return torch.from_numpy(grayscale_image)


def preprocess_batch(frames, bkg_color=np.array([144, 72, 17])):
    """
    Preprocess and stack a batch of images along a new channel dimension.

    Args:
        frames (ndarray): Batch of images with shape (batch_size, 3, 64, 84).
        bkg_color (ndarray): Background color to subtract, default is [144, 72, 17].

    Returns:
        Tensor: Batch of preprocessed images with shape (batch_size, 1, 64, 64).
    """
    batch_size = frames.shape[0]

    # Preprocess each frame in the batch
    preprocessed_frames = torch.stack(
        [preprocess(frames[i], bkg_color) for i in range(batch_size)]
    ).float()

    # Add a channel dimension to get (batch_size, 1, 64, 64)
    preprocessed_frames = preprocessed_frames.unsqueeze(1)

    return preprocessed_frames
