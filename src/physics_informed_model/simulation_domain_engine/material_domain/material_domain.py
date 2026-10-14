import torch
import torch.nn as nn
from types import SimpleNamespace
from simulation_domain_engine.material_domain.polygon_signed_distance_field import compute_polygon_signed_distance_field
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

class MaterialDomain:
    def __init__(self, domain_name="air_domain", simulation_domain=None):
        self.simulation_domain = simulation_domain
        self.domain_id = None
        self.domain_name = domain_name
        
        self.geometry_config = SimpleNamespace(
            outline=None
        )
        
        self.physics_config = SimpleNamespace(
            relative_permeability=1.0,
            coercive=[0.0, 0.0],
            current=0.0,
            bh_curve_data=None
        )
        
        self.training_config = SimpleNamespace(
            pde_weight=1.0,
            interface_weight=10.0,
            sampling_density_multiplier=1.0,
            number_of_interior_points=2000,
            number_of_surface_points=500
        )
        
        self.network_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=32,
            activation_function=nn.SiLU(),
            fourier_features=False,
            fourier_sigma=1.0,
            fourier_mapping_size=256
        )
        
        self.outline_tensor = None
        self.section_area = 0.0
        self.current_density = 0.0

    def build_domain(self):
        if self.geometry_config.outline is not None:
            self.outline_tensor = torch.tensor(self.geometry_config.outline, dtype=torch.float32)
            self._compute_section_area()
            self._compute_current_density()
        else:
            raise ValueError(f"Lỗi: Domain '{self.domain_name}' chưa được cấp outline trong geometry_config.")
            
        if self.physics_config.bh_curve_data is None:
            self._compute_constant_bh_curve()
            
        return self

    def _compute_constant_bh_curve(self):
        nu0 = self.simulation_domain.nu0 if self.simulation_domain is not None else VACUUM_RELUCTIVITY
        constant_reluctivity = nu0 / self.physics_config.relative_permeability
        self.physics_config.bh_curve_data = lambda points_tensor: torch.full((points_tensor.shape[0], 1), constant_reluctivity, dtype=torch.float32, device=points_tensor.device)
        return self.physics_config.bh_curve_data

    def _compute_section_area(self):
        if self.geometry_config.outline is None or len(self.geometry_config.outline) < 3:
            self.section_area = 0.0
            return self.section_area
        x = [p[0] for p in self.geometry_config.outline]
        y = [p[1] for p in self.geometry_config.outline]
        self.section_area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
        return self.section_area

    def _compute_current_density(self):
        area = self.section_area 
        if area > 0.0:
            self.current_density = self.physics_config.current / area
        else:
            self.current_density = 0.0
        return self.current_density

    def compute_signed_distance_field(self, points_tensor):
        if self.outline_tensor is None:
            raise RuntimeError(f"Domain '{self.domain_name}' chưa được gọi hàm build_domain().")
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(self, points_tensor):
        return self.physics_config.bh_curve_data(points_tensor)

    def evaluate_magnetization_vector(self, points_tensor):
        hx_tensor = torch.full((points_tensor.shape[0], 1), self.physics_config.coercive[0], dtype=torch.float32, device=points_tensor.device)
        hy_tensor = torch.full((points_tensor.shape[0], 1), self.physics_config.coercive[1], dtype=torch.float32, device=points_tensor.device)
        return hx_tensor, hy_tensor

    def evaluate_current_density(self, points_tensor):
        return torch.full((points_tensor.shape[0], 1), self.current_density, dtype=torch.float32, device=points_tensor.device)
