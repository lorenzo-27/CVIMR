# Quick Start Guide

Get up and running with the XOR Representation Learning project in minutes!

## Quick Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests to verify setup
python test_setup.py

# 3. Launch Jupyter notebook
jupyter notebook XOR_Representation_Learning.ipynb
```

## Quick Experiments

### Run a Single Experiment

```bash
# XOR with ReLU
python run_experiment.py --dataset xor --activation relu --hidden 2

# Two Moons with Tanh
python run_experiment.py --dataset two_moons --activation tanh --hidden 8

# Spiral with different hidden sizes
python run_experiment.py --dataset spiral --activation relu --hidden 16
```

## Project Structure
TBD

## Common Use Cases

### 1. Explore XOR Problem

```python
from CVIMR.model import TwoLayerNet, get_device
from CVIMR.data import generate_xor
from CVIMR.training import train_model
from CVIMR.visualization import plot_decision_boundary

# Setup
device = get_device()
x_data, y_data = generate_xor()

# Train
model = TwoLayerNet(input_dim=2, hidden_dim=2, output_dim=1, activation='relu')
history = train_model(model, x_data, y_data, device=device)

# Visualize
plot_decision_boundary(model, x_data, y_data, device=device)
```

### 2. Compare Activations

```python
activations = ['relu', 'tanh', 'sigmoid']
results = {}

for act in activations:
    model = TwoLayerNet(activation=act)
    history = train_model(model, x_data, y_data, device=device)
    results[act] = history
```

### 3. Test Noise Robustness

```python
from CVIMR.data import generate_xor

noise_levels = [0.0, 0.05, 0.1, 0.2]

for noise in noise_levels:
    x_noisy, y_noisy = generate_xor(noise=noise)
    model = TwoLayerNet()
    history = train_model(model, x_noisy, y_noisy, device=device)
```

### 4. Explore Complex Datasets

```python
from CVIMR.data import generate_two_moons, generate_spiral

# Two Moons
x_moons, y_moons = generate_two_moons(n_samples=100, noise=0.1)
model_moons = TwoLayerNet(input_dim=2, hidden_dim=8, activation='relu')
train_model(model_moons, x_moons, y_moons, device=device)

# Spiral
x_spiral, y_spiral = generate_spiral(n_samples=100, noise=0.2)
model_spiral = TwoLayerNet(input_dim=2, hidden_dim=8, activation='relu')
train_model(model_spiral, x_spiral, y_spiral, device=device)
```

## Visualization Examples

### Decision Boundaries

```python
from CVIMR.visualization import plot_decision_boundary

fig = plot_decision_boundary(model, x_data, y_data, device=device,
                             title="My Decision Boundary")
plt.show()
```

### Latent Space Trajectories

```python
from CVIMR.visualization import plot_latent_trajectories

latent_history = history.get_latent_tensor()
fig = plot_latent_trajectories(latent_history, y_data,
                               title="Learning Dynamics")
plt.show()
```

### Weight Evolution

```python
from CVIMR.visualization import plot_weight_trajectories

fig = plot_weight_trajectories(history,
                               title="Parameter Evolution")
plt.show()
```

### Create Animation

```python
from CVIMR.visualization import create_latent_animation

latent_history = history.get_latent_tensor()
create_latent_animation(latent_history, y_data,
                        filename="learning.gif",
                        interval=50)
```

## Configuration

### Change Training Parameters

```python
history = train_model(
    model=model,
    x_data=x_data,
    y_data=y_data,
    lr=0.05,              # Learning rate
    max_epochs=20000,     # Maximum epochs
    threshold=1e-6,       # Early stopping threshold
    device=device,
    record_interval=20,   # Record every N epochs
    use_bce=True         # Use BCE loss (classification)
)
```

### Customize Model

```python
model = TwoLayerNet(
    input_dim=2,          # Input dimension
    hidden_dim=4,         # Hidden layer size
    output_dim=1,         # Output dimension
    activation='tanh'     # relu, tanh, or sigmoid
)
```

### Generate Custom Data

```python
# XOR with noise
x_data, y_data = generate_xor(noise=0.1)

# 3-input XOR
x_data, y_data = generate_3input_xor()

# Two Moons
x_data, y_data = generate_two_moons(n_samples=200, noise=0.15)

# Spiral
x_data, y_data = generate_spiral(n_samples=150, noise=0.25)

# Circles
x_data, y_data = generate_circles(n_samples=200, noise=0.1)
```