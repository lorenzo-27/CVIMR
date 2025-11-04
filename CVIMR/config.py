"""
Configuration file for experiments.
"""

# Default hyperparameters
DEFAULT_CONFIG = {
    'learning_rate': 0.01,
    'max_epochs': 10000,
    'threshold': 1e-5,
    'hidden_dim': 2,
    'activation': 'relu',
    'use_bce': True,
    'record_interval': 10,
}

# Dataset configurations
DATASET_CONFIGS = {
    'xor': {
        'n_samples': 4,
        'noise': 0.0,
        'input_dim': 2,
        'hidden_dim': 2,
    },
    '3d_xor': {
        'n_samples': 8,
        'noise': 0.0,
        'input_dim': 3,
        'hidden_dim': 4,
    },
    'two_moons': {
        'n_samples': 1000,
        'noise': 0.1,
        'input_dim': 2,
        'hidden_dim': 8,
    },
    'spiral': {
        'n_samples': 1000,
        'noise': 0.2,
        'input_dim': 2,
        'hidden_dim': 8,
    },
    'circles': {
        'n_samples': 1000,
        'noise': 0.1,
        'input_dim': 2,
        'hidden_dim': 8,
    },
}

# Visualization settings
VIZ_CONFIG = {
    'figsize_default': (10, 8),
    'figsize_wide': (14, 6),
    'figsize_comparison': (18, 5),
    'dpi': 150,
    'decision_boundary_resolution': 200,
    'loss_landscape_resolution': 30,
    'animation_fps': 20,
    'animation_interval': 50,
}

# Color schemes
COLORS = {
    'trajectory': ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"],
    'loss_curve': 'steelblue',
    'accuracy_curve': 'green',
    'weight_colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
}

# Training display settings
DISPLAY_CONFIG = {
    'print_interval': 1000,  # Print progress every N epochs
    'progress_bar': True,
    'use_rich': True,
}

# Model architecture presets
MODEL_PRESETS = {
    'minimal': {
        'hidden_dim': 2,
        'activation': 'relu',
    },
    'small': {
        'hidden_dim': 4,
        'activation': 'relu',
    },
    'medium': {
        'hidden_dim': 8,
        'activation': 'tanh',
    },
    'large': {
        'hidden_dim': 16,
        'activation': 'relu',
    },
}

# Random seeds for reproducibility
SEEDS = {
    'default': 42,
    'experiments': [42, 123, 456, 789, 1024],
}