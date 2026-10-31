import os
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
