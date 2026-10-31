from typing import Tuple, Union, List, Optional, Any
from types import SimpleNamespace
import torch
import torch.nn as nn

from simulation_domain_engine.material_domain.polygon_signed_distance_field import compute_polygon_signed_distance_field
from simulation_domain_engine.material_domain import assign_material as _assign_material_module
from simulation_domain_engine.material_domain import assign_coercive_direction as _assign_coercive_direction_module
from simulation_domain_engine.material_domain import build_domain as _build_domain_module
from simulation_domain_engine.material_domain import compute_constant_bh_curve as _compute_constant_bh_curve_module
from simulation_domain_engine.material_domain import compute_section_area as _compute_section_area_module
from simulation_domain_engine.material_domain import compute_current_density as _compute_current_density_module
from simulation_domain_engine.material_domain import evaluate_reluctivity as _evaluate_reluctivity_module
from simulation_domain_engine.material_domain import evaluate_magnetization_vector as _evaluate_magnetization_vector_module
from simulation_domain_engine.material_domain import evaluate_current_density as _evaluate_current_density_module

class MaterialDomain:

    def __init__(self, domain_name: str = "air_domain", simulation_domain: Optional[Any] = None) -> None:
        self.simulation_domain = simulation_domain
        self.domain_id: Optional[int] = None
        self.domain_name = domain_name
        
        self.geometry_config = SimpleNamespace(
            outline=None
        )
        
        self.physics_config = SimpleNamespace(
            relative_permeability=1.0,
            coercive=[0.0, 0.0],
            current=0.0,
            bh_curve_data=None,
            is_nonlinear=False
        )
        
        self.training_config = SimpleNamespace(
            pde_weight=1.0,
            pde_nonlinear_weight=1.0,
            interface_weight=1.0,
            interface_potential_weight=1000.0,
            interface_bn_weight=500.0,
            interface_ht_weight=500.0,
            pde_fading_beta=100.0,
            sampling_density_multiplier=1.0,
            number_of_interior_points=2000,
            number_of_surface_points=1200
        )
        
        self.network_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU(),
            fourier_features=True,
            fourier_sigma=1.2,
            fourier_mapping_size=128
        )
        
        self.outline_tensor: Optional[torch.Tensor] = None
        self.section_area: float = 0.0
        self.current_density: Union[float, Any] = 0.0
        self.coercive_magnitude: float = 0.0

        self.precomputed_edge_starts: Optional[torch.Tensor] = None
        self.precomputed_edge_vecs: Optional[torch.Tensor] = None
        self.precomputed_edge_lens: Optional[torch.Tensor] = None
        self.precomputed_unit_tangents: Optional[torch.Tensor] = None
        self.precomputed_unit_normals: Optional[torch.Tensor] = None

    def assign_material(self, material_name: str = "air") -> "MaterialDomain":
        return _assign_material_module.assign_material(self, material_name=material_name)

    def assign_coercive_direction(self, direction: Tuple[float, float] = (0, 1)) -> "MaterialDomain":
        return _assign_coercive_direction_module.assign_coercive_direction(self, direction=direction)

    def build_domain(self) -> "MaterialDomain":
        return _build_domain_module.build_domain(self)

    def update_material_domain(self) -> "MaterialDomain":
        """
        Đồng bộ toàn bộ thuộc tính cấu hình trong physics_config, training_config, network_config
        và tái khởi tạo cấu trúc hình học miền vật liệu.
        """
        return self.build_domain()

    def _compute_constant_bh_curve(self) -> Any:
        return _compute_constant_bh_curve_module.compute_constant_bh_curve(self)

    def _compute_section_area(self) -> float:
        return _compute_section_area_module.compute_section_area(self)

    def _compute_current_density(self) -> Union[float, Any]:
        return _compute_current_density_module.compute_current_density(self)

    def compute_signed_distance_field(self, points_tensor: torch.Tensor) -> torch.Tensor:
        if self.outline_tensor is None:
            raise RuntimeError(f"Domain '{self.domain_name}' chưa được gọi hàm build_domain().")
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(
        self,
        points_tensor: torch.Tensor,
        B_magnitude_tensor: Optional[torch.Tensor] = None,
        is_nonlinear: Optional[bool] = None
    ) -> torch.Tensor:
        return _evaluate_reluctivity_module.evaluate_reluctivity(
            self, points_tensor, B_magnitude_tensor=B_magnitude_tensor, is_nonlinear=is_nonlinear
        )

    def evaluate_magnetization_vector(self, points_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _evaluate_magnetization_vector_module.evaluate_magnetization_vector(self, points_tensor)

    def evaluate_current_density(self, points_tensor: torch.Tensor) -> torch.Tensor:
        return _evaluate_current_density_module.evaluate_current_density(self, points_tensor)
