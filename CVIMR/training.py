import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import track

console = Console()


class TrainingHistory:
    """Store training history."""

    def __init__(self):
        self.losses = []
        self.latent_representations = []
        self.weights_history = []
        self.accuracies = []

    def record(self, loss, latent_rep, weights=None, accuracy=None):
        """Record training metrics."""
        self.losses.append(loss)
        self.latent_representations.append(latent_rep.detach().clone())
        if weights is not None:
            self.weights_history.append(weights)
        if accuracy is not None:
            self.accuracies.append(accuracy)

    def get_latent_tensor(self):
        """Get latent representations as tensor [epochs, samples, dims]."""
        return torch.stack(self.latent_representations, dim=0)


def train_model(model, x_data, y_data, lr=0.1, max_epochs=10000,
                threshold=1e-5, device='cpu', record_interval=1,
                use_bce=True):
    """
    Train the neural network.

    Args:
        model: Neural network model
        x_data: Input data
        y_data: Target data
        lr: Learning rate
        max_epochs: Maximum number of epochs
        threshold: Loss threshold for early stopping
        device: Device to train on
        record_interval: Interval for recording history
        use_bce: Use BCE loss instead of MSE

    Returns:
        TrainingHistory object
    """
    model = model.to(device)
    x_data = x_data.to(device)
    y_data = y_data.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Use BCE loss for classification
    if use_bce:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    history = TrainingHistory()
    loss_val = float('inf')
    epoch = 0

    console.print(f"[bold green]Training on {device}...[/bold green]")

    # Training loop with progress bar
    for epoch in track(range(max_epochs), description="Training"):
        # Forward pass
        output, hidden_rep = model(x_data)
        loss = criterion(output, y_data)
        loss_val = loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            if use_bce:
                predictions = torch.sigmoid(output) > 0.5
            else:
                predictions = output > 0.5
            accuracy = (predictions == y_data).float().mean().item()

        # Record history
        if epoch % record_interval == 0 or loss_val <= threshold:
            weights = {
                'hidden_weight': model.hidden_layer.weight.detach().clone().cpu(),
                'hidden_bias': model.hidden_layer.bias.detach().clone().cpu(),
                'output_weight': model.output_layer.weight.detach().clone().cpu(),
                'output_bias': model.output_layer.bias.detach().clone().cpu()
            }
            history.record(loss_val, hidden_rep.cpu(), weights, accuracy)

        # Early stopping
        if loss_val <= threshold:
            break

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            console.print(f"Epoch {epoch}: Loss {loss_val:.6f}, Accuracy {accuracy:.4f}")

    console.print(f"[bold green]Training finished after {epoch + 1} epochs[/bold green]")
    console.print(f"Final loss: {loss_val:.6f}")

    return history


def compute_accuracy(model, x_data, y_data, device='cpu', use_sigmoid=True):
    """
    Compute model accuracy.

    Args:
        model: Neural network model
        x_data: Input data
        y_data: Target data
        device: Device to compute on
        use_sigmoid: Apply sigmoid to output

    Returns:
        float: Accuracy value
    """
    model.eval()
    with torch.no_grad():
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        output, _ = model(x_data)

        if use_sigmoid:
            predictions = torch.sigmoid(output) > 0.5
        else:
            predictions = output > 0.5

        accuracy = (predictions == y_data).float().mean().item()

    return accuracy