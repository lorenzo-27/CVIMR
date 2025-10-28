#!/usr/bin/env python3
"""
Quick experiment runner for testing different configurations.
"""
import argparse
import torch
from rich.console import Console
from rich.table import Table
from pathlib import Path

from .model import TwoLayerNet, get_device
from .data import (generate_xor, generate_3input_xor,
                        generate_two_moons, generate_spiral, generate_circles)
from .training import train_model, compute_accuracy
from .visualization import plot_decision_boundary, plot_latent_trajectories

console = Console()

DATASETS = {
    'xor': generate_xor,
    '3d_xor': generate_3input_xor,
    'two_moons': lambda: generate_two_moons(n_samples=100, noise=0.1),
    'spiral': lambda: generate_spiral(n_samples=100, noise=0.2),
    'circles': lambda: generate_circles(n_samples=100, noise=0.1),
}


def main():
    parser = argparse.ArgumentParser(description='Run neural network experiments')
    parser.add_argument('--dataset', type=str, default='xor',
                        choices=list(DATASETS.keys()),
                        help='Dataset to use')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh', 'sigmoid'],
                        help='Activation function')
    parser.add_argument('--hidden', type=int, default=2,
                        help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Maximum epochs')
    parser.add_argument('--threshold', type=float, default=1e-5,
                        help='Loss threshold for early stopping')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Input noise level')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create output directories
    plots_dir = Path('plots')
    checkpoints_dir = Path('checkpoints')
    plots_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    # Get device
    device = get_device()

    # Print configuration
    console.print("\n[bold cyan]═══ Experiment Configuration ═══[/bold cyan]")

    table = Table(show_header=False, box=None)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Dataset", args.dataset)
    table.add_row("Activation", args.activation)
    table.add_row("Hidden Size", str(args.hidden))
    table.add_row("Learning Rate", str(args.lr))
    table.add_row("Max Epochs", str(args.epochs))
    table.add_row("Device", str(device))
    table.add_row("Random Seed", str(args.seed))

    console.print(table)
    console.print()

    # Generate data
    console.print("[bold]Generating dataset...[/bold]")
    data_generator = DATASETS[args.dataset]

    if args.dataset == 'xor' and args.noise > 0:
        x_data, y_data = generate_xor(noise=args.noise)
    else:
        x_data, y_data = data_generator()

    input_dim = x_data.shape[1]
    console.print(f"Data shape: X={x_data.shape}, Y={y_data.shape}")

    # Create model
    console.print("\n[bold]Creating model...[/bold]")
    model = TwoLayerNet(
        input_dim=input_dim,
        hidden_dim=args.hidden,
        output_dim=1,
        activation=args.activation
    )

    # Try identity initialization for 2x2 case
    if args.hidden == 2 and input_dim == 2:
        try:
            model.initialize_identity()
            console.print("✓ Initialized with identity weights")
        except:
            pass

    console.print(f"Model: {model}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"Total parameters: {n_params}")

    # Train model
    console.print("\n[bold cyan]═══ Training ═══[/bold cyan]")
    history = train_model(
        model=model,
        x_data=x_data,
        y_data=y_data,
        lr=args.lr,
        max_epochs=args.epochs,
        threshold=args.threshold,
        device=device,
        record_interval=max(1, args.epochs // 100),
        use_bce=True
    )

    # Evaluate
    console.print("\n[bold cyan]═══ Evaluation ═══[/bold cyan]")
    accuracy = compute_accuracy(model, x_data, y_data, device=device)
    final_loss = history.losses[-1]

    results_table = Table(show_header=False, box=None)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Final Loss", f"{final_loss:.6f}")
    results_table.add_row("Accuracy", f"{accuracy:.4f} ({accuracy * 100:.2f}%)")
    results_table.add_row("Epochs Trained", str(len(history.losses)))

    console.print(results_table)

    # Test predictions
    console.print("\n[bold]Sample Predictions:[/bold]")
    model.eval()
    with torch.no_grad():
        outputs, _ = model(x_data.to(device))
        predictions = (torch.sigmoid(outputs) > 0.5).cpu().float()

        pred_table = Table()
        pred_table.add_column("Sample", style="cyan")
        pred_table.add_column("Input", style="white")
        pred_table.add_column("True", style="yellow")
        pred_table.add_column("Pred", style="green")
        pred_table.add_column("Match", style="bold")

        for i in range(min(8, len(x_data))):
            input_str = str(x_data[i].numpy().round(2))
            true_val = str(int(y_data[i].item()))
            pred_val = str(int(predictions[i].item()))
            match = "✓" if true_val == pred_val else "✗"
            match_style = "green" if match == "✓" else "red"

            pred_table.add_row(
                f"{i}",
                input_str,
                true_val,
                pred_val,
                f"[{match_style}]{match}[/{match_style}]"
            )

        console.print(pred_table)

    # Visualizations
    if not args.no_plot:
        console.print("\n[bold cyan]═══ Generating Visualizations ═══[/bold cyan]")

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Decision boundary (only for 2D inputs)
        if input_dim == 2:
            console.print("Plotting decision boundary...")
            fig = plot_decision_boundary(
                model, x_data, y_data,
                device=device,
                title=f"{args.dataset.upper()} - {args.activation.upper()} Decision Boundary"
            )
            filename = plots_dir / f"decision_boundary_{args.dataset}_{args.activation}-model_{args.dataset}_{args.activation}_h{args.hidden}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            console.print(f"✓ Saved to {filename}")

        # Latent trajectories (only for 2D hidden layer)
        if args.hidden == 2:
            console.print("Plotting latent trajectories...")
            latent_history = history.get_latent_tensor()
            fig = plot_latent_trajectories(
                latent_history, y_data,
                title=f"{args.dataset.upper()} - {args.activation.upper()} Latent Trajectories"
            )
            filename = plots_dir / f"latent_trajectories_{args.dataset}_{args.activation}-model_{args.dataset}_{args.activation}_h{args.hidden}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            console.print(f"✓ Saved to {filename}")

        # Loss curve
        console.print("Plotting loss curve...")
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(len(history.losses))
        ax.plot(epochs, history.losses, linewidth=2, color='steelblue')
        ax.set_yscale('log')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.set_title(f"{args.dataset.upper()} - Training Loss",
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        filename = plots_dir / f"loss_curve_{args.dataset}_{args.activation}-model_{args.dataset}_{args.activation}_h{args.hidden}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        console.print(f"✓ Saved to {filename}")

        # Accuracy curve (if available)
        if history.accuracies:
            console.print("Plotting accuracy curve...")
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(len(history.accuracies))
            ax.plot(epochs, history.accuracies, linewidth=2, color='green')
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(f"{args.dataset.upper()} - Training Accuracy",
                         fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            filename = plots_dir / f"accuracy_curve_{args.dataset}_{args.activation}-model_{args.dataset}_{args.activation}_h{args.hidden}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            console.print(f"✓ Saved to {filename}")

    # Save model
    model_filename = checkpoints_dir / f"model_{args.dataset}_{args.activation}_h{args.hidden}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden,
            'output_dim': 1,
            'activation': args.activation
        },
        'history': {
            'losses': history.losses,
            'accuracies': history.accuracies
        },
        'args': vars(args)
    }, model_filename)

    console.print(f"\n✓ Model saved to {model_filename}")

    # Summary
    console.print("\n[bold green]═══ Experiment Complete! ═══[/bold green]")

    if accuracy == 1.0:
        console.print("[bold green]🎉 Perfect accuracy achieved![/bold green]")
    elif accuracy >= 0.9:
        console.print("[bold yellow]⚡ High accuracy achieved![/bold yellow]")
    else:
        console.print("[bold red]⚠️  Consider adjusting hyperparameters[/bold red]")


if __name__ == '__main__':
    main()