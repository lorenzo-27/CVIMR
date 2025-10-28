import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    """Two-layer neural network with configurable activation function."""

    def __init__(self, input_dim=2, hidden_dim=2, output_dim=1, activation='relu'):
        """
        Initialize the network.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(TwoLayerNet, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Select activation function
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if activation.lower() not in activation_functions:
            raise ValueError(f"Activation must be one of {list(activation_functions.keys())}")

        self.activation = activation_functions[activation.lower()]
        self.activation_name = activation.lower()

    def forward(self, x):
        """
        Forward pass.

        Returns:
            tuple: (output, hidden_activation)
        """
        z_hidden = self.hidden_layer(x)
        a_hidden = self.activation(z_hidden)
        out = self.output_layer(a_hidden)
        return out, a_hidden

    def initialize_identity(self):
        """Initialize hidden layer as identity transform (for 2x2 case)."""
        with torch.no_grad():
            if self.hidden_layer.weight.shape == (2, 2):
                self.hidden_layer.weight.copy_(torch.eye(2))
                self.hidden_layer.bias.fill_(0.)
            else:
                raise ValueError("Identity initialization only supported for 2x2 hidden layer")


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')