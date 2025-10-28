# Exploring Learned Representations with XOR

A comprehensive project for visualizing and understanding how neural networks learn internal representations through various experiments on XOR and other classification tasks.

## Project Structure
TBD

## Features

### Implemented Experiments

1. **Activation Function Comparison**
   - Train networks with ReLU, Tanh, and Sigmoid activations
   - Visualize how different activations affect learned representations
   - Compare decision boundaries and convergence speed

2. **Latent Space Trajectories**
   - Track how data points move in hidden space during training
   - Visualize the evolution of internal representations
   - Create animations of the learning process

3. **Decision Boundaries**
   - High-resolution visualization of model predictions
   - Compare boundaries across different architectures
   - Understand how networks separate classes

4. **Weight Trajectories**
   - Track how network parameters evolve during training
   - Visualize weight and bias changes over epochs
   - Correlate parameter changes with loss reduction

5. **Activation Histograms**
   - Analyze neuron activation distributions
   - Compare activations at different training stages
   - Identify dead neurons and saturation issues

6. **Noise Robustness**
   - Add input noise and observe stability in hidden space
   - Test network resilience to perturbations
   - Understand generalization capabilities

7. **3-Input XOR with PCA**
   - Visualize higher-dimensional hidden representations
   - Use PCA for dimensionality reduction
   - Explore complex latent spaces

8. **Complex Datasets**
   - Two Moons dataset
   - Spiral dataset
   - Concentric Circles dataset
   - Demonstrates scalability to non-trivial problems

9. **Loss Landscape Visualization**
   - 3D visualization of the optimization surface
   - Understand the difficulty of the learning problem
   - Visualize local minima and saddle points

10. **Training Loss Analysis**
    - Track loss reduction over epochs
    - Correlate loss drops with representation changes
    - Compare convergence across different settings

## Installation

TBD

## Usage

### Running the Notebook

```bash
jupyter notebook XOR_Representation_Learning.ipynb
```

Or use JupyterLab:

```bash
jupyter lab
```

### Using Individual Modules

You can also import and use the modules separately:

```python
from CVIMR.model import TwoLayerNet, get_device
from CVIMR.data import generate_xor, generate_two_moons
from CVIMR.training import train_model
from CVIMR.visualization import plot_decision_boundary

# Get device (CPU/CUDA/MPS)
device = get_device()

# Generate data
x_data, y_data = generate_xor()

# Create and train model
model = TwoLayerNet(input_dim=2, hidden_dim=2, output_dim=1, activation='relu')
history = train_model(model, x_data, y_data, device=device)

# Visualize
plot_decision_boundary(model, x_data, y_data, device=device)
```

## Key Components

### model.py

- `TwoLayerNet`: Flexible two-layer neural network
  - Supports ReLU, Tanh, and Sigmoid activations
  - Optional identity initialization
  - Returns both output and hidden activations

- `get_device()`: Automatically selects best available device (CUDA > MPS > CPU)

### data.py

Dataset generators for various tasks:
- `generate_xor()`: Classic 2-input XOR
- `generate_3input_xor()`: 3-input XOR for higher dimensions
- `generate_two_moons()`: Non-linearly separable moons
- `generate_spiral()`: Interleaved spiral patterns
- `generate_circles()`: Concentric circles

All support noise injection for robustness testing.

### training.py

- `train_model()`: Comprehensive training loop
  - Supports BCE and MSE loss
  - Early stopping
  - Progress tracking with Rich
  - Records history (loss, accuracy, weights, activations)

- `TrainingHistory`: Data structure for storing training metrics

### visualization.py

Extensive visualization toolkit:
- `plot_latent_trajectories()`: 2D trajectory visualization
- `plot_decision_boundary()`: High-res decision boundaries
- `plot_loss_curve()`: Training loss over epochs
- `plot_weight_trajectories()`: Parameter evolution
- `plot_activation_histograms()`: Neuron activation analysis
- `plot_3d_latent_pca()`: 3D PCA visualization
- `create_latent_animation()`: Animated learning process

## Device Support

The code automatically detects and uses the best available device:

- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon
- **CPU**: Fallback for systems without GPU acceleration

All tensors and models are automatically moved to the appropriate device.

## References
TBD

## License
TBD
