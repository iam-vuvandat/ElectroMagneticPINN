import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def compute_theoretical_upper_bound(in_dim, num_layers, width):
    bound_per_layer = sum(math.comb(width, j) for j in range(min(in_dim, width) + 1))
    return bound_per_layer**num_layers


class ReLUNetwork(nn.Module):
    def __init__(self, in_dim, num_layers, width):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def get_activation_patterns(self, x):
        patterns = []
        curr = x
        for layer in self.net:
            curr = layer(curr)
            if isinstance(layer, nn.ReLU):
                patterns.append((curr > 0).to(torch.int8))
        return torch.cat(patterns, dim=1)


def main():
    in_dim = 2
    num_layers = 2
    width = 2
    num_samples_per_dim = 500

    theoretical_max = compute_theoretical_upper_bound(in_dim, num_layers, width)
    print(f"Input Dimension: {in_dim}")
    print(f"Hidden Layers: {num_layers}")
    print(f"Neurons per Layer: {width}")
    print(f"Theoretical Upper Bound Regions: {theoretical_max:,}")

    torch.manual_seed(42)
    model = ReLUNetwork(in_dim, num_layers, width)

    if in_dim == 1:
        x_vals = torch.linspace(-2.0, 2.0, num_samples_per_dim).unsqueeze(1)
        with torch.no_grad():
            patterns = model.get_activation_patterns(x_vals)
            y_vals = model.net(x_vals)

        unique_patterns = torch.unique(patterns, dim=0)
        empirical_count = unique_patterns.shape[0]
        print(f"Empirical Discovered Regions in domain [-2, 2]: {empirical_count:,}")

        plt.figure(figsize=(10, 5))
        plt.plot(x_vals.numpy(), y_vals.numpy(), label="Network Output")
        plt.title(f"1D Piecewise Linear Output ({num_layers}x{width} ReLU Net - {empirical_count} Regions)")
        plt.xlabel("Input X")
        plt.ylabel("Output Y")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif in_dim == 2:
        x_range = torch.linspace(-2.0, 2.0, num_samples_per_dim)
        y_range = torch.linspace(-2.0, 2.0, num_samples_per_dim)
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="ij")
        grid_pts = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)

        with torch.no_grad():
            patterns = model.get_activation_patterns(grid_pts)

        unique_patterns, pattern_indices = torch.unique(patterns, dim=0, return_inverse=True)
        empirical_count = unique_patterns.shape[0]
        print(f"Empirical Discovered Regions in domain [-2, 2]^2: {empirical_count:,}")

        region_map = pattern_indices.reshape(num_samples_per_dim, num_samples_per_dim).numpy()

        plt.figure(figsize=(8, 7))
        plt.imshow(region_map, extent=[-2.0, 2.0, -2.0, 2.0], origin="lower", cmap="prism")
        plt.colorbar(label="Region Index")
        plt.title(f"2D Piecewise Linear Partition ({num_layers}x{width} ReLU Net - {empirical_count} Regions)")
        plt.xlabel("Input X1")
        plt.ylabel("Input X2")
        plt.show()


if __name__ == "__main__":
    main()