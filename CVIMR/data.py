import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles


def generate_xor(n_samples=4, noise=0.0):
    """
    Generate XOR dataset.

    Args:
        n_samples: Number of samples (4 for canonical XOR)
        noise: Standard deviation of Gaussian noise to add

    Returns:
        tuple: (x_data, y_data) as torch tensors
    """
    x_data = torch.tensor([[0., 0.],
                           [0., 1.],
                           [1., 0.],
                           [1., 1.]])
    y_data = torch.tensor([[0.],
                           [1.],
                           [1.],
                           [0.]])

    if noise > 0:
        x_data += torch.randn_like(x_data) * noise

    return x_data, y_data


def generate_3input_xor(n_samples=8, noise=0.0):
    """
    Generate 3-input XOR dataset.

    Args:
        n_samples: Number of samples (8 for canonical 3-input XOR)
        noise: Standard deviation of Gaussian noise to add

    Returns:
        tuple: (x_data, y_data) as torch tensors
    """
    x_data = torch.tensor([[0., 0., 0.],
                           [0., 0., 1.],
                           [0., 1., 0.],
                           [0., 1., 1.],
                           [1., 0., 0.],
                           [1., 0., 1.],
                           [1., 1., 0.],
                           [1., 1., 1.]])

    # XOR of all three inputs
    y_data = (x_data[:, 0].long() ^ x_data[:, 1].long() ^ x_data[:, 2].long()).float().unsqueeze(1)

    if noise > 0:
        x_data += torch.randn_like(x_data) * noise

    return x_data, y_data


def generate_two_moons(n_samples=1000, noise=0.1):
    """
    Generate two moons dataset.

    Args:
        n_samples: Number of samples
        noise: Standard deviation of Gaussian noise

    Returns:
        tuple: (x_data, y_data) as torch tensors
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)


def generate_spiral(n_samples=1000, noise=0.1):
    """
    Generate spiral dataset.

    Args:
        n_samples: Number of samples per class
        noise: Standard deviation of Gaussian noise

    Returns:
        tuple: (x_data, y_data) as torch tensors
    """
    n = n_samples
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi  # Random angle

    # Spiral 1
    r1 = 2 * theta + np.pi
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)

    # Spiral 2
    r2 = -2 * theta - np.pi
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)

    # Combine
    X = np.vstack([np.c_[x1, y1], np.c_[x2, y2]])
    y = np.hstack([np.zeros(n), np.ones(n)])

    # Add noise
    X += np.random.randn(*X.shape) * noise

    # Normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)


def generate_circles(n_samples=1000, noise=0.1):
    """
    Generate concentric circles dataset.

    Args:
        n_samples: Number of samples
        noise: Standard deviation of Gaussian noise

    Returns:
        tuple: (x_data, y_data) as torch tensors
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)