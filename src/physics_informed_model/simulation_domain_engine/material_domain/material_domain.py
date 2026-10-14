import torch
import torch.nn as nn
from types import SimpleNamespace
from simulation_domain_engine.material_domain.polygon_signed_distance_field import compute_polygon_signed_distance_field

VACUUM_RELUCTIVITY = 795774.715459

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
            interface_weight=1.0,
            interface_potential_weight=1000.0,
            interface_bn_weight=500.0,
            interface_ht_weight=500.0,
            pde_fading_beta=150.0,
            sampling_density_multiplier=1.0,
            number_of_interior_points=1000,
            number_of_surface_points=800
        )
        
        self.network_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU(),
            fourier_features=True,
            fourier_sigma=1.2,
            fourier_mapping_size=128
        )
        
        self.outline_tensor = None
        self.section_area = 0.0
        self.current_density = 0.0

    def build_domain(self):
        if self.geometry_config.outline is not None:
            if len(self.geometry_config.outline) < 3:
                raise ValueError(f"Lỗi: Outline của domain '{self.domain_name}' phải có ít nhất 3 đỉnh.")
            self.outline_tensor = torch.tensor(self.geometry_config.outline, dtype=torch.float32)
            self._compute_section_area()
            if self.section_area <= 1e-12:
                raise ValueError(f"Lỗi: Diện tích của domain '{self.domain_name}' bằng 0 hoặc suy biến.")
            self._compute_current_density()
        else:
            raise ValueError(f"Lỗi: Domain '{self.domain_name}' chưa được cấp outline trong geometry_config.")
            
        if self.physics_config.bh_curve_data is None:
            self._compute_constant_bh_curve()
            
        return self

    def _compute_constant_bh_curve(self):
        def _default_reluctivity_fn(points_tensor, B_magnitude_tensor=None):
            const_nu = VACUUM_RELUCTIVITY / float(self.physics_config.relative_permeability)
            return torch.full((points_tensor.shape[0], 1), const_nu, dtype=points_tensor.dtype, device=points_tensor.device)
        self.physics_config.bh_curve_data = _default_reluctivity_fn
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
        if callable(self.physics_config.current):
            self.current_density = self.physics_config.current
            return self.current_density
        area = self.section_area 
        if area > 0.0:
            self.current_density = float(self.physics_config.current) / area
        else:
            self.current_density = 0.0
        return self.current_density

    def compute_signed_distance_field(self, points_tensor):
        if self.outline_tensor is None:
            raise RuntimeError(f"Domain '{self.domain_name}' chưa được gọi hàm build_domain().")
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(self, points_tensor, B_magnitude_tensor=None):
        if self.physics_config.bh_curve_data is None:
            self._compute_constant_bh_curve()
        try:
            return self.physics_config.bh_curve_data(points_tensor, B_magnitude_tensor)
        except TypeError:
            return self.physics_config.bh_curve_data(points_tensor)

    def evaluate_magnetization_vector(self, points_tensor):
        coercive_cfg = self.physics_config.coercive
        if callable(coercive_cfg):
            res = coercive_cfg(points_tensor)
            return res[:, 0:1], res[:, 1:2]
        if isinstance(coercive_cfg, torch.Tensor):
            c_tensor = coercive_cfg.to(device=points_tensor.device, dtype=points_tensor.dtype)
            if c_tensor.dim() == 1:
                hx = torch.full((points_tensor.shape[0], 1), c_tensor[0].item(), dtype=points_tensor.dtype, device=points_tensor.device)
                hy = torch.full((points_tensor.shape[0], 1), c_tensor[1].item(), dtype=points_tensor.dtype, device=points_tensor.device)
                return hx, hy
            return c_tensor[:, 0:1], c_tensor[:, 1:2]
        hx_tensor = torch.full((points_tensor.shape[0], 1), float(coercive_cfg[0]), dtype=points_tensor.dtype, device=points_tensor.device)
        hy_tensor = torch.full((points_tensor.shape[0], 1), float(coercive_cfg[1]), dtype=points_tensor.dtype, device=points_tensor.device)
        return hx_tensor, hy_tensor

    def evaluate_current_density(self, points_tensor):
        if callable(self.current_density):
            val = self.current_density(points_tensor)
            return val.view(-1, 1).to(device=points_tensor.device, dtype=points_tensor.dtype)
        return torch.full((points_tensor.shape[0], 1), float(self.current_density), dtype=points_tensor.dtype, device=points_tensor.device)
