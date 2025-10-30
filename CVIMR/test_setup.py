#!/usr/bin/env python3
"""
Test setup and verify all components work correctly.
"""
import sys
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def test_imports():
    """Test if all required modules can be imported."""
    console.print("\n[bold cyan]Testing imports...[/bold cyan]")

    modules = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('rich', 'Rich'),
    ]

    results = []
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            results.append((display_name, "✓", "green"))
        except ImportError as e:
            results.append((display_name, f"✗ {str(e)}", "red"))

    table = Table(show_header=False)
    table.add_column("Module", style="cyan")
    table.add_column("Status")

    for name, status, color in results:
        table.add_row(name, f"[{color}]{status}[/{color}]")

    console.print(table)

    # Check if all passed
    all_passed = all(status == "✓" for _, status, _ in results)
    return all_passed


def test_custom_modules():
    """Test custom modules."""
    console.print("\n[bold cyan]Testing custom modules...[/bold cyan]")

    try:
        from CVIMR.model import TwoLayerNet, get_device
        from CVIMR.data import generate_xor, generate_two_moons
        from CVIMR.training import train_model, TrainingHistory
        from CVIMR.visualization import plot_decision_boundary

        console.print("[green]✓ All custom modules imported successfully[/green]")
        return True
    except ImportError as e:
        console.print(f"[red]✗ Failed to import custom modules: {e}[/red]")
        return False


def test_device():
    """Test device detection."""
    console.print("\n[bold cyan]Testing device detection...[/bold cyan]")

    try:
        from CVIMR.model import get_device
        device = get_device()

        table = Table(show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Device", str(device))
        table.add_row("CUDA available", str(torch.cuda.is_available()))
        table.add_row("MPS available", str(torch.backends.mps.is_available()))

        if torch.cuda.is_available():
            table.add_row("CUDA device count", str(torch.cuda.device_count()))
            table.add_row("CUDA device name", torch.cuda.get_device_name(0))

        console.print(table)
        return True
    except Exception as e:
        console.print(f"[red]✗ Device detection failed: {e}[/red]")
        return False


def test_model_creation():
    """Test model creation and initialization."""
    console.print("\n[bold cyan]Testing model creation...[/bold cyan]")

    try:
        from CVIMR.model import TwoLayerNet

        activations = ['relu', 'tanh', 'sigmoid']

        for activation in activations:
            model = TwoLayerNet(input_dim=2, hidden_dim=2, output_dim=1,
                                activation=activation)
            n_params = sum(p.numel() for p in model.parameters())
            console.print(f"✓ {activation.upper()}: {n_params} parameters")

        # Test identity initialization
        model = TwoLayerNet(input_dim=2, hidden_dim=2, output_dim=1, activation='relu')
        model.initialize_identity()
        console.print("✓ Identity initialization works")

        return True
    except Exception as e:
        console.print(f"[red]✗ Model creation failed: {e}[/red]")
        return False


def test_data_generation():
    """Test data generation."""
    console.print("\n[bold cyan]Testing data generation...[/bold cyan]")

    try:
        from CVIMR.data import (generate_xor, generate_3input_xor,
                                generate_two_moons, generate_spiral, generate_circles)

        datasets = {
            'XOR': generate_xor,
            '3D XOR': generate_3input_xor,
            'Two Moons': lambda: generate_two_moons(n_samples=50),
            'Spiral': lambda: generate_spiral(n_samples=50),
            'Circles': lambda: generate_circles(n_samples=50),
        }

        table = Table()
        table.add_column("Dataset", style="cyan")
        table.add_column("Shape", style="green")
        table.add_column("Labels", style="yellow")

        for name, generator in datasets.items():
            x_data, y_data = generator()
            unique_labels = len(torch.unique(y_data))
            table.add_row(name, str(x_data.shape), str(unique_labels))

        console.print(table)
        return True
    except Exception as e:
        console.print(f"[red]✗ Data generation failed: {e}[/red]")
        return False


def test_training():
    """Test training pipeline."""
    console.print("\n[bold cyan]Testing training pipeline...[/bold cyan]")

    try:
        from CVIMR.model import TwoLayerNet, get_device
        from CVIMR.data import generate_xor
        from training import train_model

        device = get_device()
        x_data, y_data = generate_xor()

        model = TwoLayerNet(input_dim=2, hidden_dim=2, output_dim=1, activation='relu')

        console.print("Training for 100 epochs (quick test)...")
        history = train_model(
            model=model,
            x_data=x_data,
            y_data=y_data,
            lr=0.1,
            max_epochs=100,
            threshold=1e-10,  # Won't reach threshold
            device=device,
            record_interval=10,
            use_bce=True
        )

        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Epochs trained", str(len(history.losses)))
        table.add_row("Initial loss", f"{history.losses[0]:.6f}")
        table.add_row("Final loss", f"{history.losses[-1]:.6f}")
        table.add_row("Loss reduced", f"{history.losses[0] - history.losses[-1]:.6f}")

        console.print(table)

        if history.losses[-1] < history.losses[0]:
            console.print("[green]✓ Training reduces loss[/green]")
            return True
        else:
            console.print("[red]✗ Training did not reduce loss[/red]")
            return False

    except Exception as e:
        console.print(f"[red]✗ Training failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_visualization():
    """Test visualization functions."""
    console.print("\n[bold cyan]Testing visualization...[/bold cyan]")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        from CVIMR.model import TwoLayerNet, get_device
        from CVIMR.data import generate_xor
        from training import train_model
        from visualization import (plot_decision_boundary, plot_latent_trajectories,
                                   plot_loss_curve)

        device = get_device()
        x_data, y_data = generate_xor()

        model = TwoLayerNet(input_dim=2, hidden_dim=2, output_dim=1, activation='relu')
        history = train_model(
            model=model,
            x_data=x_data,
            y_data=y_data,
            lr=0.1,
            max_epochs=100,
            threshold=1e-10,
            device=device,
            record_interval=10,
            use_bce=True
        )

        # Test each visualization
        tests = []

        # Decision boundary
        try:
            fig = plot_decision_boundary(model, x_data, y_data, device=device)
            plt.close(fig)
            tests.append(("Decision Boundary", "✓", "green"))
        except Exception as e:
            tests.append(("Decision Boundary", f"✗ {e}", "red"))

        # Latent trajectories
        try:
            latent_history = history.get_latent_tensor()
            fig = plot_latent_trajectories(latent_history, y_data)
            plt.close(fig)
            tests.append(("Latent Trajectories", "✓", "green"))
        except Exception as e:
            tests.append(("Latent Trajectories", f"✗ {e}", "red"))

        # Loss curve
        try:
            fig = plot_loss_curve(history)
            plt.close(fig)
            tests.append(("Loss Curve", "✓", "green"))
        except Exception as e:
            tests.append(("Loss Curve", f"✗ {e}", "red"))

        table = Table(show_header=False)
        table.add_column("Visualization", style="cyan")
        table.add_column("Status")

        for name, status, color in tests:
            table.add_row(name, f"[{color}]{status}[/{color}]")

        console.print(table)

        all_passed = all(status == "✓" for _, status, _ in tests)
        return all_passed

    except Exception as e:
        console.print(f"[red]✗ Visualization testing failed: {e}[/red]")
        return False


def test_full_experiment():
    """Run a complete mini experiment."""
    console.print("\n[bold cyan]Running complete mini experiment...[/bold cyan]")

    try:
        from CVIMR.model import TwoLayerNet, get_device
        from CVIMR.data import generate_xor
        from training import train_model, compute_accuracy

        device = get_device()
        x_data, y_data = generate_xor()

        console.print("Training XOR with ReLU activation...")
        model = TwoLayerNet(input_dim=2, hidden_dim=2, output_dim=1, activation='relu')
        model.initialize_identity()

        history = train_model(
            model=model,
            x_data=x_data,
            y_data=y_data,
            lr=0.1,
            max_epochs=5000,
            threshold=1e-5,
            device=device,
            record_interval=50,
            use_bce=True
        )

        accuracy = compute_accuracy(model, x_data, y_data, device=device)

        table = Table(title="Experiment Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Final Loss", f"{history.losses[-1]:.6f}")
        table.add_row("Accuracy", f"{accuracy:.4f} ({accuracy * 100:.2f}%)")
        table.add_row("Epochs", str(len(history.losses)))
        table.add_row("Converged", "✓" if history.losses[-1] < 1e-3 else "✗")

        console.print(table)

        # Test predictions
        model.eval()
        with torch.no_grad():
            outputs, _ = model(x_data.to(device))
            predictions = (torch.sigmoid(outputs) > 0.5).cpu().float()

            pred_table = Table(title="Predictions")
            pred_table.add_column("Input", style="white")
            pred_table.add_column("True", style="yellow")
            pred_table.add_column("Pred", style="green")
            pred_table.add_column("✓", style="bold")

            for i in range(len(x_data)):
                input_str = str(x_data[i].numpy())
                true_val = str(int(y_data[i].item()))
                pred_val = str(int(predictions[i].item()))
                match = "✓" if true_val == pred_val else "✗"
                color = "green" if match == "✓" else "red"

                pred_table.add_row(
                    input_str,
                    true_val,
                    pred_val,
                    f"[{color}]{match}[/{color}]"
                )

            console.print(pred_table)

        return accuracy >= 0.9  # Consider successful if >90% accuracy

    except Exception as e:
        console.print(f"[red]✗ Full experiment failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold cyan]XOR Representation Learning - Setup Test[/bold cyan]\n"
        "Testing all components...",
        border_style="cyan"
    ))

    tests = [
        ("Import Dependencies", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Device Detection", test_device),
        ("Model Creation", test_model_creation),
        ("Data Generation", test_data_generation),
        ("Training Pipeline", test_training),
        ("Visualization", test_visualization),
        ("Full Experiment", test_full_experiment),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            console.print(f"[red]Unexpected error in {name}: {e}[/red]")
            results.append((name, False))

    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        "[bold cyan]Test Summary[/bold cyan]",
        border_style="cyan"
    ))

    summary_table = Table()
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", style="bold")

    for name, passed in results:
        status = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
        summary_table.add_row(name, status)

    console.print(summary_table)

    # Final verdict
    all_passed = all(passed for _, passed in results)
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    console.print()
    if all_passed:
        console.print(Panel.fit(
            f"[bold green]🎉 All tests passed! ({passed_count}/{total_count})[/bold green]\n"
            "[green]Setup is complete and ready to use.[/green]",
            border_style="green"
        ))
        sys.exit(0)
    else:
        console.print(Panel.fit(
            f"[bold red]⚠️ Some tests failed ({passed_count}/{total_count} passed)[/bold red]\n"
            "[red]Please check the errors above and install missing dependencies.[/red]",
            border_style="red"
        ))
        sys.exit(1)


if __name__ == '__main__':
    main()