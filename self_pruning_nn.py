"""
Self-Pruning Neural Network with Learnable Sigmoid Gates and L1 Sparsity Regularization

This script implements a self-pruning feed-forward classifier on CIFAR-10 using
a custom `PrunableLinear` layer with per-weight learnable gates and an L1 sparsity penalty.
"""

import os
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with learnable, per-weight sigmoid gates.

    Forward:
        gates = sigmoid(gate_scores)
        pruned_weight = weight * gates
        output = input @ pruned_weight.T + bias

    Gradients naturally flow through both `weight` and `gate_scores` because
    every operation is standard PyTorch autograd-tracked math (no custom backward).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Classification weight (Kaiming uniform init, matching nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Gate scores, initialized to 0 so sigmoid(0) = 0.5 (all connections half-open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / (fan_in**0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Convenience accessor used by the sparsity loss and eval code."""
        return torch.sigmoid(self.gate_scores)


class SelfPruningNet(nn.Module):
    """Feed-forward classifier for CIFAR-10 built from PrunableLinear layers."""

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)

        self.hidden_layers = nn.ModuleList([
            PrunableLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.output_layer = PrunableLinear(dims[-1], num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten (B, C, H, W) -> (B, D)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        return self.output_layer(x)

    def prunable_layers(self) -> List[PrunableLinear]:
        """All PrunableLinear layers (used by the sparsity loss / stats code)."""
        return list(self.hidden_layers) + [self.output_layer]


def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of every gate value in every PrunableLinear layer.
    Gates are always in (0, 1), so this is simply their sum.
    Minimizing this pushes gate_scores toward -inf, i.e. gates -> 0,
    which is exactly what 'pruning a connection' means here.
    """
    device = next(model.parameters()).device
    total = torch.tensor(0.0, device=device)
    for layer in model.prunable_layers():
        total = total + layer.get_gates().sum()
    return total


def get_dataloaders(
    batch_size: int = 256,
    data_root: str = "./data",
    num_workers: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Downloads CIFAR-10 and creates training and testing dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def build_optimizer(
    model: SelfPruningNet,
    weight_lr: float = 1e-3,
    gate_lr: float = 1e-2,
) -> torch.optim.Optimizer:
    """
    Two-learning-rate optimizer: weights and gates are placed in separate
    Adam parameter groups so gates can saturate toward 0/1 within fewer epochs
    without needing a destabilizingly large learning rate on classification weights.
    """
    weight_params, gate_params = [], []
    for layer in model.prunable_layers():
        weight_params.append(layer.weight)
        if layer.bias is not None:
            weight_params.append(layer.bias)
        gate_params.append(layer.gate_scores)

    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": weight_lr, "name": "weights"},
        {"params": gate_params, "lr": gate_lr, "name": "gates"},
    ])
    return optimizer


def train_model(
    model: SelfPruningNet,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    lam: float,
    weight_lr: float = 1e-3,
    gate_lr: float = 1e-2,
    epochs: int = 15,
    log_every: int = 200,
) -> Tuple[SelfPruningNet, Dict[str, List[float]]]:
    """Trains one SelfPruningNet with sparsity coefficient `lam`."""
    model = model.to(device)
    optimizer = build_optimizer(model, weight_lr, gate_lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "cls_loss": [], "sparsity_loss": []}

    for epoch in range(epochs):
        model.train()
        running_total, running_cls, running_sp = 0.0, 0.0, 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            cls_loss = ce_loss_fn(outputs, labels)
            sp_loss = sparsity_loss(model)
            total_loss = cls_loss + lam * sp_loss

            total_loss.backward()
            optimizer.step()

            running_total += total_loss.item()
            running_cls += cls_loss.item()
            running_sp += sp_loss.item()

            if (i + 1) % log_every == 0:
                print(
                    f"  [lambda={lam}] epoch {epoch+1}/{epochs} "
                    f"batch {i+1}/{len(train_loader)} "
                    f"total={total_loss.item():.4f} "
                    f"cls={cls_loss.item():.4f} "
                    f"sparsity_raw={sp_loss.item():.1f}"
                )

        n_batches = len(train_loader)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(running_total / n_batches)
        history["cls_loss"].append(running_cls / n_batches)
        history["sparsity_loss"].append(running_sp / n_batches)

    return model, history


@torch.no_grad()
def evaluate_accuracy(
    model: SelfPruningNet,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Computes classification top-1 accuracy on the test set."""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_sparsity(
    model: SelfPruningNet,
    threshold: float = 1e-2,
) -> Tuple[float, np.ndarray]:
    """Percentage of gate values below `threshold`, across every PrunableLinear layer."""
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.get_gates().flatten())
    all_gates = torch.cat(all_gates)
    sparsity_pct = 100.0 * (all_gates < threshold).float().mean().item()
    return sparsity_pct, all_gates.cpu().numpy()


def plot_gate_distribution(
    gate_values: np.ndarray,
    lam: float,
    threshold: float = 1e-2,
    output_path: str = "gate_distribution.png",
):
    """Plots and saves the gate value histogram zoomed into [0.0, 0.2]."""
    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=100, color="#4C72B0", edgecolor="black", alpha=0.8)
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"prune threshold = {threshold}"
    )
    plt.xlim(0.0, 0.2)
    plt.title(f"Distribution of Gate Values (lambda = {lam})")
    plt.xlabel("Gate value = sigmoid(gate_scores)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved gate distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Self-Pruning Neural Network on CIFAR-10."
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[1e-6, 1e-5, 1e-3],
        help="Sparsity regularization coefficients (e.g. 1e-6 1e-5 1e-3).",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="DataLoader batch size.")
    parser.add_argument("--weight-lr", type=float, default=1e-3, help="Learning rate for weights.")
    parser.add_argument("--gate-lr", type=float, default=1e-2, help="Learning rate for gate scores.")
    parser.add_argument("--threshold", type=float, default=1e-2, help="Pruning threshold for gates.")
    parser.add_argument("--data-root", type=str, default="./data", help="Root folder for CIFAR-10.")
    parser.add_argument(
        "--plot-output",
        type=str,
        default="gate_distribution.png",
        help="Path to save gate distribution histogram.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Sanity check autograd on PrunableLinear
    sanity_layer = PrunableLinear(4, 3)
    x = torch.randn(2, 4, requires_grad=True)
    out = sanity_layer(x).sum()
    out.backward()
    assert sanity_layer.weight.grad is not None, "Weight gradient check failed!"
    assert sanity_layer.gate_scores.grad is not None, "Gate scores gradient check failed!"
    print("Sanity check passed: Gradients flow to both weights and gate scores.")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, data_root=args.data_root
    )

    results = []
    trained_models = {}
    gate_distributions = {}

    for lam in args.lambdas:
        print(f"\n===== Training with lambda = {lam} =====")
        model = SelfPruningNet()
        model, _ = train_model(
            model,
            train_loader,
            device=device,
            lam=lam,
            weight_lr=args.weight_lr,
            gate_lr=args.gate_lr,
            epochs=args.epochs,
        )

        acc = evaluate_accuracy(model, test_loader, device=device)
        sparsity_pct, gate_vals = evaluate_sparsity(model, threshold=args.threshold)

        print(f"Result: lambda={lam} | Test Accuracy: {acc:.2f}% | Sparsity: {sparsity_pct:.2f}%")
        results.append({
            "Lambda": lam,
            "Test Accuracy (%)": round(acc, 2),
            "Sparsity Level (%)": round(sparsity_pct, 2),
        })
        trained_models[lam] = model
        gate_distributions[lam] = gate_vals

    df = pd.DataFrame(results)
    print("\n===== Summary of Experiment Results =====")
    print(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df)

    # Plot for the medium lambda (sweet spot)
    target_lam = 1e-5 if 1e-5 in gate_distributions else args.lambdas[0]
    plot_gate_distribution(
        gate_distributions[target_lam],
        lam=target_lam,
        threshold=args.threshold,
        output_path=args.plot_output,
    )


if __name__ == "__main__":
    main()
