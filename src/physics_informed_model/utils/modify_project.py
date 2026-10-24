import os
import shutil

def reorganize_simulation_domain_structure():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    sim_engine_dir = os.path.join(project_root, 'simulation_domain_engine')
    mat_domain_dir = os.path.join(sim_engine_dir, 'material_domain')
    sim_methods_dir = os.path.join(sim_engine_dir, 'simulation_domain_methods')
    tests_dir = os.path.join(project_root, 'tests')

    os.makedirs(sim_engine_dir, exist_ok=True)
    os.makedirs(mat_domain_dir, exist_ok=True)
    os.makedirs(tests_dir, exist_ok=True)

    # 1. DỌN DẸP THƯ MỤC CON DƯ THỪA & FILE LỖI THỜI
    files_to_remove = [
        os.path.join(sim_engine_dir, 'add_material_domain.py')
    ]
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[*] Đã xóa tệp thừa: {os.path.relpath(file_path, project_root)}")

    if os.path.exists(sim_methods_dir):
        shutil.rmtree(sim_methods_dir)
        print(f"[*] Đã xóa thư mục con dư thừa: {os.path.relpath(sim_methods_dir, project_root)}")

    files_to_create = {}

    # -------------------------------------------------------------------------
    # 2. TẠO CÁC HÀM PHỤ TRỢ ĐẶT NGAY TRONG simulation_domain_engine/
    # -------------------------------------------------------------------------

    # 2.1 set_boundary.py
    files_to_create[os.path.join(sim_engine_dir, 'set_boundary.py')] = '''def set_boundary(self, boundary_object):
    boundary_object.simulation_domain = self
    self.boundary = boundary_object
    return self
'''

    # 2.2 validate_non_overlapping_domains.py
    files_to_create[os.path.join(sim_engine_dir, 'validate_non_overlapping_domains.py')] = '''import torch

def validate_non_overlapping_domains(self, candidate_domain=None, check_resolution=80):
    if getattr(self.physics_config, 'allow_overlapping_domains', False):
        return True
    mat_domains = [d for d in self.domains_list if getattr(d, 'domain_name', '') != 'background']
    if candidate_domain is not None and getattr(candidate_domain, 'domain_name', '') != 'background':
        if candidate_domain not in mat_domains:
            mat_domains.append(candidate_domain)
    if len(mat_domains) < 2:
        return True

    x_min, x_max = self.x_boundaries_tuple
    y_min, y_max = self.y_boundaries_tuple
    xs = torch.linspace(x_min, x_max, check_resolution)
    ys = torch.linspace(y_min, y_max, check_resolution)
    Xg, Yg = torch.meshgrid(xs, ys, indexing='ij')
    test_pts = torch.column_stack([Xg.reshape(-1), Yg.reshape(-1)])

    strict_inside_counts = torch.zeros(test_pts.shape[0], dtype=torch.int32)
    for d in mat_domains:
        sdf = d.compute_signed_distance_field(test_pts).view(-1)
        strict_inside_counts += (sdf < -1e-6).int()

    if torch.any(strict_inside_counts > 1):
        raise ValueError("Lỗi hình học: Phát hiện các miền vật liệu chồng lấn lên nhau.")
    return True
'''

    # 2.3 add_material_domain_method.py
    files_to_create[os.path.join(sim_engine_dir, 'add_material_domain_method.py')] = '''def add_material_domain(self, domain_object):
    domain_object.simulation_domain = self
    if domain_object.outline_tensor is None:
        domain_object.build_domain()

    if getattr(domain_object, 'domain_name', '') != 'background':
        self.validate_non_overlapping_domains(candidate_domain=domain_object)

    if self.background_domain in self.domains_list:
        insert_idx = len(self.domains_list) - 1
        self.domains_list.insert(insert_idx, domain_object)
    else:
        self.domains_list.append(domain_object)
        
    for idx, dom in enumerate(self.domains_list):
        dom.domain_id = idx + 1
    return self
'''

    # 2.4 create_material_domain.py
    files_to_create[os.path.join(sim_engine_dir, 'create_material_domain.py')] = '''from simulation_domain_engine.material_domain.material_domain import MaterialDomain

def create_material_domain(
    self,
    domain_name: str,
    material_name: str = None,
    outline: list = None,
    coercive_direction: tuple = None
) -> MaterialDomain:
    domain = MaterialDomain(domain_name=domain_name, simulation_domain=self)
    if outline is not None:
        domain.geometry_config.outline = outline
    if material_name is not None:
        domain.assign_material(material_name)
    if coercive_direction is not None:
        domain.assign_coercive_direction(coercive_direction)
    self.add_material_domain(domain)
    return domain
'''

    # 2.5 build_background_domain.py
    files_to_create[os.path.join(sim_engine_dir, 'build_background_domain.py')] = '''from simulation_domain_engine.material_domain.material_domain import MaterialDomain

def build_background_domain(self, domain_name="background", training_config=None, network_config=None):
    if self.background_domain in self.domains_list:
        self.domains_list.remove(self.background_domain)

    x_min, x_max = self.boundary.geometry_config.x_boundaries_tuple
    y_min, y_max = self.boundary.geometry_config.y_boundaries_tuple
    
    box_outline = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]

    bg_domain = MaterialDomain(domain_name=domain_name, simulation_domain=self)
    bg_domain.geometry_config.outline = box_outline
    bg_domain.physics_config.relative_permeability = 1.0
    bg_domain.physics_config.coercive = [0.0, 0.0]
    bg_domain.physics_config.current = 0.0
    bg_domain.training_config.number_of_interior_points = 4000
    bg_domain.training_config.number_of_surface_points = 0

    if training_config is not None:
        for key, value in vars(training_config).items():
            setattr(bg_domain.training_config, key, value)

    if network_config is not None:
        for key, value in vars(network_config).items():
            setattr(bg_domain.network_config, key, value)

    bg_domain.build_domain()

    total_box_area = (x_max - x_min) * (y_max - y_min)
    total_sub_material_area = sum(d.section_area for d in self.domains_list)
    bg_domain.section_area = max(0.0, total_box_area - total_sub_material_area)

    self.background_domain = bg_domain
    self.domains_list.append(self.background_domain)

    for idx, dom in enumerate(self.domains_list):
        dom.domain_id = idx + 1

    return self.background_domain
'''

    # 2.6 get_domain_count.py
    files_to_create[os.path.join(sim_engine_dir, 'get_domain_count.py')] = '''def get_domain_count(self):
    return len(self.domains_list)
'''

    # 2.7 display_methods.py
    files_to_create[os.path.join(sim_engine_dir, 'display_methods.py')] = '''from simulation_domain_engine.domain_visualizer import display_simulation_domain

def plot_problem_definition(self, resolution=100):
    display_simulation_domain(self, self.x_boundaries_tuple, self.y_boundaries_tuple, resolution)

def display(self, x_boundaries_tuple=None, y_boundaries_tuple=None, resolution=150):
    xb = x_boundaries_tuple if x_boundaries_tuple is not None else self.x_boundaries_tuple
    yb = y_boundaries_tuple if y_boundaries_tuple is not None else self.y_boundaries_tuple
    display_simulation_domain(self, xb, yb, resolution)
'''

    # -------------------------------------------------------------------------
    # 3. FILE MẸ SIMULATION_DOMAIN IMPORT TỪ CÙNG THƯ MỤC GỐC
    # -------------------------------------------------------------------------
    files_to_create[os.path.join(sim_engine_dir, 'simulation_domain.py')] = '''from typing import List, Tuple, Dict, Any, Optional
from types import SimpleNamespace
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

    @property
    def A0(self) -> float:
        return (self.H0 * self.L0) / self.nu0

    @property
    def B0(self) -> float:
        return self.H0 / self.nu0

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
'''

    # Ghi lại toàn bộ các file đã tạo/cập nhật
    for file_path, content in files_to_create.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + '\n')
        print(f"[*] Đã tạo/cập nhật: {os.path.relpath(file_path, project_root)}")

    print("\n[✔] TẢI LẠI CẤU TRÚC HOÀN TẤT: Tất cả hàm phụ trợ của SimulationDomain hiện nằm trực tiếp trong thư mục 'simulation_domain_engine/'!")

if __name__ == "__main__":
    reorganize_simulation_domain_structure()