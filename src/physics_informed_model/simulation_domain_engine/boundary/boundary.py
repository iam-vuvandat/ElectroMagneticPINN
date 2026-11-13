import torch
import numpy as np
from types import SimpleNamespace
from simulation_domain_engine.boundary.build_boundary import build_boundary
from simulation_domain_engine.boundary.compute_boundary_factor import compute_boundary_factor

class Boundary:
    build_boundary = build_boundary
    compute_boundary_factor = compute_boundary_factor

    def __init__(self, simulation_domain=None, boundary_name="outer_boundary", boundary_type="dirichlet"):
        self.simulation_domain = simulation_domain
        self.boundary_name = boundary_name
        self.boundary_type = boundary_type

        self.geometry_config = SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05),
            shape="box"
        )

        self.physics_config = SimpleNamespace(
            potential_value=0.0,
            enforce_method="hard_factor"
        )

        self.training_config = SimpleNamespace(
            boundary_loss_weight=100.0,
            number_of_boundary_points=1000
        )

        self.bounding_box = None
        self.outline_tensor = None

    def sample_boundary_points(self, device=torch.device('cpu')):
        n_pts = int(self.training_config.number_of_boundary_points)
        if n_pts <= 0 or self.outline_tensor is None:
            return torch.empty((0, 2), dtype=torch.float32, device=device)

        outline = self.outline_tensor.cpu().numpy()
        num_vertices = len(outline)
        edge_starts = outline
        edge_ends = np.roll(edge_starts, shift=-1, axis=0)
        edge_vectors = edge_ends - edge_starts
        edge_lengths = np.sqrt(np.sum(edge_vectors ** 2, axis=1))
        total_perimeter = np.sum(edge_lengths)
        edge_probs = edge_lengths / max(total_perimeter, 1e-12)

        sampled_edges = np.random.choice(num_vertices, size=n_pts, p=edge_probs)
        t_params = np.random.rand(n_pts, 1)
        pts_sampled = edge_starts[sampled_edges] + t_params * edge_vectors[sampled_edges]
        return torch.tensor(pts_sampled, dtype=torch.float32, device=device, requires_grad=True)
