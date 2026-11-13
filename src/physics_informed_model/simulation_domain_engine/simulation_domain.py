from typing import List, Tuple, Dict, Any, Optional
from types import SimpleNamespace
from functools import cached_property
import torch

from simulation_domain_engine.boundary.boundary import Boundary
from simulation_domain_engine.material_domain.material_domain import MaterialDomain
from simulation_domain_engine.material_library.material_data import MaterialData
from simulation_domain_engine.global_signed_distance_field import compute_global_signed_distance_field
from simulation_domain_engine.global_physical_properties_evaluation import (
    evaluate_global_physical_properties,
    VACUUM_RELUCTIVITY
)

from simulation_domain_engine import set_boundary as _set_boundary_fn
from simulation_domain_engine import validate_non_overlapping_domains as _validate_domains_fn
from simulation_domain_engine import add_material_domain_method as _add_material_domain_fn
from simulation_domain_engine import create_material_domain as _create_material_domain_fn
from simulation_domain_engine import build_background_domain as _build_bg_domain_fn
from simulation_domain_engine import get_domain_count as _get_domain_count_fn
from simulation_domain_engine import display_methods as _display_methods_module


class SimulationDomain:

    def __init__(
        self,
        electro_magnetic_pinn: Optional[Any] = None,
        x_boundaries_tuple: Tuple[float, float] = (-0.05, 0.05),
        y_boundaries_tuple: Tuple[float, float] = (-0.05, 0.05)
    ) -> None:
        self.electro_magnetic_pinn = electro_magnetic_pinn
        self.material_data = MaterialData()

        self.physics_config = SimpleNamespace(
            scale_L0=0.1,
            scale_H0=800000.0,
            scale_nu0=VACUUM_RELUCTIVITY,
            allow_overlapping_domains=False
        )

        self.boundary = Boundary(simulation_domain=self)
        self.boundary.geometry_config.x_boundaries_tuple = x_boundaries_tuple
        self.boundary.geometry_config.y_boundaries_tuple = y_boundaries_tuple
        self.boundary.build_boundary()

        self.domains_list: List[MaterialDomain] = []
        self.background_domain: Optional[MaterialDomain] = None

    @property
    def L0(self) -> float:
        return float(self.physics_config.scale_L0)

    @property
    def H0(self) -> float:
        return float(self.physics_config.scale_H0)

    @property
    def nu0(self) -> float:
        return float(self.physics_config.scale_nu0)

    @cached_property
    def A0(self) -> float:
        return (self.H0 * self.L0) / self.nu0

    @cached_property
    def B0(self) -> float:
        return self.H0 / self.nu0

    def clear_scale_cache(self) -> None:
        self.__dict__.pop('A0', None)
        self.__dict__.pop('B0', None)

    @property
    def x_boundaries_tuple(self) -> Tuple[float, float]:
        return self.boundary.geometry_config.x_boundaries_tuple

    @x_boundaries_tuple.setter
    def x_boundaries_tuple(self, bounds: Tuple[float, float]) -> None:
        self.boundary.geometry_config.x_boundaries_tuple = bounds
        self.boundary.build_boundary()

    @property
    def y_boundaries_tuple(self) -> Tuple[float, float]:
        return self.boundary.geometry_config.y_boundaries_tuple

    @y_boundaries_tuple.setter
    def y_boundaries_tuple(self, bounds: Tuple[float, float]) -> None:
        self.boundary.geometry_config.y_boundaries_tuple = bounds
        self.boundary.build_boundary()

    def set_boundary(self, boundary_object: Boundary) -> "SimulationDomain":
        return _set_boundary_fn.set_boundary(self, boundary_object)

    def validate_non_overlapping_domains(
        self,
        candidate_domain: Optional[MaterialDomain] = None,
        check_resolution: int = 80
    ) -> bool:
        return _validate_domains_fn.validate_non_overlapping_domains(
            self, candidate_domain=candidate_domain, check_resolution=check_resolution
        )

    def add_material_domain(self, domain_object: MaterialDomain) -> "SimulationDomain":
        return _add_material_domain_fn.add_material_domain(self, domain_object)

    def create_material_domain(
        self,
        domain_name: str,
        material_name: Optional[str] = None,
        outline: Optional[list] = None,
        coercive_direction: Optional[Tuple[float, float]] = None
    ) -> MaterialDomain:
        return _create_material_domain_fn.create_material_domain(
            self,
            domain_name=domain_name,
            material_name=material_name,
            outline=outline,
            coercive_direction=coercive_direction
        )

    def build_background_domain(
        self,
        domain_name: str = "background",
        training_config: Optional[Any] = None,
        network_config: Optional[Any] = None
    ) -> MaterialDomain:
        return _build_bg_domain_fn.build_background_domain(
            self,
            domain_name=domain_name,
            training_config=training_config,
            network_config=network_config
        )

    def get_domain_count(self) -> int:
        return _get_domain_count_fn.get_domain_count(self)

    def compute_global_signed_distance_field(self, points_tensor: torch.Tensor) -> torch.Tensor:
        return compute_global_signed_distance_field(self.domains_list, points_tensor)

    def evaluate_global_physical_properties(self, points_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return evaluate_global_physical_properties(self.domains_list, points_tensor)

    def plot_problem_definition(self, resolution: int = 100) -> None:
        _display_methods_module.plot_problem_definition(self, resolution=resolution)

    def display(
        self,
        x_boundaries_tuple: Optional[Tuple[float, float]] = None,
        y_boundaries_tuple: Optional[Tuple[float, float]] = None,
        resolution: int = 150
    ) -> None:
        _display_methods_module.display(
            self,
            x_boundaries_tuple=x_boundaries_tuple,
            y_boundaries_tuple=y_boundaries_tuple,
            resolution=resolution
        )
