import os

def update_project_config_architecture():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    mat_domain_file = os.path.join(
        project_root, 'simulation_domain_engine', 'material_domain', 'material_domain.py'
    )
    test_sim_file = os.path.join(
        project_root, 'tests', 'test_simulation.py'
    )

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: material_domain.py (Thêm update_material_domain)
    # =========================================================================
    files_to_update[mat_domain_file] = '''from typing import Tuple, Union, List, Optional, Any
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
'''

    # =========================================================================
    # 2. CẬP NHẬT: test_simulation.py (Lưu Config trước -> build_domain, Hội tụ sâu)
    # =========================================================================
    files_to_update[test_sim_file] = '''import os
import sys
import shutil
from types import SimpleNamespace

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from electro_magnetic_pinn import ElectroMagneticPINN

def run_linear_simulation_test():
    output_dir = "figure_linear_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN THUẦN TUYẾN TÍNH - HỘI TỤ SÂU (Adam + L-BFGS)...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. Cấu hình chuẩn hóa trường vật lý
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    # 2. Cấu hình Lấy mẫu động
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # 3. CHẾ ĐỘ THUẦN TUYẾN TÍNH - CẤU HÌNH HỘI TỤ SÂU
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = False
    model.training_config.enable_lbfgs_refinement_stage = True

    model.training_config.number_of_adam_epochs = 3500
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-9

    model.training_config.number_of_lbfgs_iterations = 1200
    model.training_config.lbfgs_learning_rate = 0.8
    model.training_config.lbfgs_maximum_evaluations = 2000
    model.training_config.lbfgs_tolerance_gradient = 1e-12
    model.training_config.lbfgs_tolerance_change = 1e-12
    model.training_config.lbfgs_history_size = 100

    # 4. Trực quan hóa
    model.visualization_config.active = True
    model.visualization_config.update_interval = 250
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80

    # -------------------------------------------------------------------------
    # 1. Ách sắt (Iron Yoke - mu_r = 1000)
    # -------------------------------------------------------------------------
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name="steel_1008",
        outline=[
            [-0.06, -0.03], 
            [0.06, -0.03], 
            [0.06, 0.0], 
            [-0.06, 0.0]
        ]
    )
    iron_yoke.physics_config.is_nonlinear = False
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_weight = 1.5
    iron_yoke.training_config.interface_potential_weight = 1000.0
    iron_yoke.training_config.interface_bn_weight = 500.0
    iron_yoke.training_config.interface_ht_weight = 500.0
    iron_yoke.training_config.pde_fading_beta = 100.0
    iron_yoke.training_config.number_of_interior_points = 4000
    iron_yoke.training_config.number_of_surface_points = 2000

    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128
    iron_yoke.build_domain()

    # -------------------------------------------------------------------------
    # 2. Nam châm trái (N30UH - mu_r = 1.05)
    # -------------------------------------------------------------------------
    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left",
        material_name="N30UH",
        outline=[
            [-0.05, 0.0], 
            [-0.005, 0.0], 
            [-0.005, 0.018], 
            [-0.05, 0.018]
        ],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_weight = 1.5
    magnet_left.training_config.interface_potential_weight = 1000.0
    magnet_left.training_config.interface_bn_weight = 500.0
    magnet_left.training_config.interface_ht_weight = 500.0
    magnet_left.training_config.pde_fading_beta = 100.0
    magnet_left.training_config.number_of_interior_points = 2500
    magnet_left.training_config.number_of_surface_points = 1500

    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128
    magnet_left.build_domain()

    # -------------------------------------------------------------------------
    # 3. Nam châm phải (N30UH - mu_r = 1.05)
    # -------------------------------------------------------------------------
    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right",
        material_name="N30UH",
        outline=[
            [0.005, 0.0], 
            [0.05, 0.0], 
            [0.05, 0.018], 
            [0.005, 0.018]
        ],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_weight = 1.5
    magnet_right.training_config.interface_potential_weight = 1000.0
    magnet_right.training_config.interface_bn_weight = 500.0
    magnet_right.training_config.interface_ht_weight = 500.0
    magnet_right.training_config.pde_fading_beta = 100.0
    magnet_right.training_config.number_of_interior_points = 2500
    magnet_right.training_config.number_of_surface_points = 1500

    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128
    magnet_right.build_domain()

    # -------------------------------------------------------------------------
    # 4. Background Air Domain
    # -------------------------------------------------------------------------
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        interface_weight=1.5,
        pde_fading_beta=100.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=10000,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=64,
        activation_function=nn.SiLU(),
        fourier_features=True,
        fourier_sigma=1.2,
        fourier_mapping_size=128
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    model.update_electromagnetic_pinn()
    sim_domain.plot_problem_definition(resolution=100)
    
    print("[*] Chạy tiến trình huấn luyện tuyến tính hội tụ sâu...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ...")
    model.plot_fields(
        resolution=180, 
        save_path=f"{output_dir}/linear_final_fields.png"
    )

    print(f"[✔] HOÀN THÀNH KIỂM THỬ TUYẾN TÍNH HỘI TỤ SÂU! Kết quả lưu tại {output_dir}/")

def test_simulation4():
    run_linear_simulation_test()

if __name__ == "__main__":
    run_linear_simulation_test()
'''

    # Thực thi cập nhật các tệp tin
    for path, code in files_to_update.items():
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code.strip() + '\n')
        print(f"[✔] Đã cập nhật tệp: {os.path.relpath(path, project_root)}")

if __name__ == "__main__":
    update_project_config_architecture()