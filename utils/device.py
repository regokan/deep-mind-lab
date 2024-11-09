import torch


def get_device() -> torch.device:
    """
    Returns the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The best available device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
