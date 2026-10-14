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
