import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import seaborn as sns


def plot_latent_trajectories(latent_history, y_data, title="Latent Space Trajectories"):
    """
    Plot trajectories of data points in latent space.

    Args:
        latent_history: Tensor of shape [epochs, samples, dims]
        y_data: Labels for coloring
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["red", "blue", "green", "orange"]

    n_samples = latent_history.shape[1]

    for i in range(n_samples):
        coords = latent_history[:, i, :]
        label_val = int(y_data[i].item())

        # Plot trajectory
        ax.plot(coords[:, 0], coords[:, 1], color=colors[i],
                alpha=0.4, linewidth=2)

        # Plot starting position (diamond)
        ax.scatter(coords[0, 0], coords[0, 1], color=colors[i],
                   marker='D', s=100, edgecolors='black',
                   linewidths=1.5, alpha=1.0, zorder=5)

        # Plot final position
        ax.scatter(coords[-1, 0], coords[-1, 1], color=colors[i],
                   alpha=0.8, marker='o', s=150,
                   label=f"Point {i}, Label={label_val}")

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Hidden Dim 1", fontsize=12)
    ax.set_ylabel("Hidden Dim 2", fontsize=12)
    ax.axis("equal")
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_decision_boundary(model, x_data, y_data, device='cpu',
                           resolution=100, title="Decision Boundary"):
    """
    Visualize decision boundary of the model.

    Args:
        model: Trained model
        x_data: Input data
        y_data: Labels
        device: Device to compute on
        resolution: Grid resolution
        title: Plot title
    """
    model.eval()

    # Create mesh grid
    x_min, x_max = x_data[:, 0].min() - 0.5, x_data[:, 0].max() + 0.5
    y_min, y_max = x_data[:, 1].min() - 0.5, x_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))

    # Predict on grid
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    with torch.no_grad():
        predictions, _ = model(grid_points)
        predictions = torch.sigmoid(predictions).cpu().numpy()

    predictions = predictions.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(xx, yy, predictions, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.contour(xx, yy, predictions, levels=[0.5], colors='black', linewidths=2)

    # Plot data points
    x_np = x_data.cpu().numpy()
    y_np = y_data.cpu().numpy().ravel()

    scatter = ax.scatter(x_np[:, 0], x_np[:, 1], c=y_np,
                         cmap='RdYlBu', s=200, edgecolors='black',
                         linewidths=2, alpha=0.9)

    plt.colorbar(contour, ax=ax, label='Prediction')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Input Dim 1", fontsize=12)
    ax.set_ylabel("Input Dim 2", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_curve(history, title="Training Loss"):
    """
    Plot training loss curve.

    Args:
        history: TrainingHistory object
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(len(history.losses))
    ax.plot(epochs, history.losses, linewidth=2, color='steelblue')
    ax.set_yscale('log')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_weight_trajectories(history, title="Weight Trajectories"):
    """
    Plot how weights change during training.

    Args:
        history: TrainingHistory object with weights_history
        title: Plot title
    """
    if not history.weights_history:
        print("No weight history recorded")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract weight evolution
    hidden_w = [w['hidden_weight'].numpy() for w in history.weights_history]
    hidden_b = [w['hidden_bias'].numpy() for w in history.weights_history]
    output_w = [w['output_weight'].numpy() for w in history.weights_history]
    output_b = [w['output_bias'].numpy() for w in history.weights_history]

    epochs = range(len(hidden_w))

    # Hidden layer weights
    ax = axes[0, 0]
    hidden_w_arr = np.array(hidden_w)
    for i in range(hidden_w_arr.shape[1]):
        for j in range(hidden_w_arr.shape[2]):
            ax.plot(epochs, hidden_w_arr[:, i, j], label=f'W[{i},{j}]')
    ax.set_title("Hidden Layer Weights", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hidden layer biases
    ax = axes[0, 1]
    hidden_b_arr = np.array(hidden_b)
    for i in range(hidden_b_arr.shape[1]):
        ax.plot(epochs, hidden_b_arr[:, i], label=f'b[{i}]')
    ax.set_title("Hidden Layer Biases", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bias Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Output layer weights
    ax = axes[1, 0]
    output_w_arr = np.array(output_w)
    for i in range(output_w_arr.shape[1]):
        for j in range(output_w_arr.shape[2]):
            ax.plot(epochs, output_w_arr[:, i, j], label=f'W[{i},{j}]')
    ax.set_title("Output Layer Weights", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Output layer biases
    ax = axes[1, 1]
    output_b_arr = np.array(output_b)
    for i in range(output_b_arr.shape[1]):
        ax.plot(epochs, output_b_arr[:, i], label=f'b[{i}]')
    ax.set_title("Output Layer Biases", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bias Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_activation_histograms(latent_history, title="Activation Histograms"):
    """
    Plot histograms of neuron activations at different stages.

    Args:
        latent_history: Tensor of shape [epochs, samples, dims]
        title: Plot title
    """
    # Sample at beginning, middle, and end
    n_epochs = latent_history.shape[0]
    stages = {
        'Beginning': 0,
        'Middle': n_epochs // 2,
        'End': -1
    }

    n_dims = latent_history.shape[2]
    fig, axes = plt.subplots(n_dims, len(stages), figsize=(15, 5 * n_dims))

    if n_dims == 1:
        axes = axes.reshape(1, -1)

    for col, (stage_name, idx) in enumerate(stages.items()):
        activations = latent_history[idx].numpy()

        for dim in range(n_dims):
            ax = axes[dim, col]
            ax.hist(activations[:, dim], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f"{stage_name} - Neuron {dim}", fontweight='bold')
            ax.set_xlabel("Activation")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_3d_latent_pca(latent_history, y_data, title="3D Latent Space (PCA)"):
    """
    Plot 3D latent space using PCA for dimensionality reduction.

    Args:
        latent_history: Tensor of shape [epochs, samples, dims]
        y_data: Labels
        title: Plot title
    """
    # Get final latent representations
    final_latent = latent_history[-1].numpy()

    # Apply PCA if needed
    if final_latent.shape[1] > 3:
        pca = PCA(n_components=3)
        latent_3d = pca.fit_transform(final_latent)
        explained_var = pca.explained_variance_ratio_
    else:
        latent_3d = final_latent
        if final_latent.shape[1] < 3:
            # Pad with zeros
            latent_3d = np.pad(latent_3d, ((0, 0), (0, 3 - final_latent.shape[1])))
        explained_var = [1.0, 0.0, 0.0]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    y_np = y_data.numpy().ravel()

    for i, point in enumerate(latent_3d):
        color = colors[i % len(colors)]
        label_val = int(y_np[i])
        ax.scatter(point[0], point[1], point[2],
                   c=color, s=200, alpha=0.8,
                   edgecolors='black', linewidths=2,
                   label=f"Point {i}, Label={label_val}")

    ax.set_xlabel(f"PC1 ({explained_var[0]:.2%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({explained_var[1]:.2%})", fontsize=12)
    ax.set_zlabel(f"PC3 ({explained_var[2]:.2%})", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    return fig


def create_latent_animation(latent_history, y_data, filename="latent_evolution.mp4",
                            interval=500, title="Latent Space Evolution"):
    """
    Create animation of latent space evolution.

    Args:
        latent_history: Tensor of shape [epochs, samples, dims]
        y_data: Labels
        filename: Output filename
        interval: Interval between frames (ms)
        title: Animation title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["red", "blue", "green", "orange"]

    n_epochs = latent_history.shape[0]
    n_samples = latent_history.shape[1]

    # Initialize scatter plots
    scatters = []
    for i in range(n_samples):
        label_val = int(y_data[i].item())
        scatter = ax.scatter([], [], color=colors[i], s=200,
                             alpha=0.8, edgecolors='black', linewidths=2,
                             label=f"Point {i}, Label={label_val}")
        scatters.append(scatter)

    ax.set_xlim(latent_history[:, :, 0].min() - 0.5,
                latent_history[:, :, 0].max() + 0.5)
    ax.set_ylim(latent_history[:, :, 1].min() - 0.5,
                latent_history[:, :, 1].max() + 0.5)
    ax.set_xlabel("Hidden Dim 1", fontsize=12)
    ax.set_ylabel("Hidden Dim 2", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    epoch_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def animate(frame):
        epoch_text.set_text(f'Epoch: {frame}')
        for i, scatter in enumerate(scatters):
            coords = latent_history[frame, i, :].numpy()
            scatter.set_offsets(coords.reshape(1, -1))
        return scatters + [epoch_text]

    anim = FuncAnimation(fig, animate, frames=n_epochs,
                         interval=interval, blit=True, repeat=True)

    anim.save(filename, writer='ffmpeg', fps=20)
    plt.close()

    print(f"Animation saved to {filename}")
    return anim