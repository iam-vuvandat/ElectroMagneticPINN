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

def run_modern_simulation_test():
    output_dir = "figure_modern_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN với API Cấu hình mới (Linear Warm-up Only)...")

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

    # 3. Cấu hình Bật/Tắt Giai đoạn & Thuật toán Tối ưu
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = False
    model.training_config.enable_lbfgs_refinement_stage = False

    model.training_config.number_of_adam_epochs = 1000
    model.training_config.linear_warmup_epochs_ratio = 1.0
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-7
    model.training_config.number_of_lbfgs_iterations = 0

    # 4. Cấu hình Trực quan hóa
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "modern_training.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "modern_training.mp4"

    def apply_standard_config(domain, int_pts=2000, surf_pts=1200, pde_nonlin_w=1.0):
        domain.training_config.pde_weight = 1.0
        domain.training_config.pde_nonlinear_weight = pde_nonlin_w
        domain.training_config.interface_weight = 1.0
        domain.training_config.interface_potential_weight = 1000.0
        domain.training_config.interface_bn_weight = 400.0
        domain.training_config.interface_ht_weight = 400.0
        domain.training_config.pde_fading_beta = 80.0
        domain.training_config.number_of_interior_points = int_pts
        domain.training_config.number_of_surface_points = surf_pts
        
        domain.network_config.hidden_layers = 4
        domain.network_config.hidden_neurons = 64
        domain.network_config.activation_function = nn.SiLU()
        domain.network_config.fourier_features = True
        domain.network_config.fourier_sigma = 1.2
        domain.network_config.fourier_mapping_size = 128
        return domain.build_domain()

    # 1. Khung Ách sắt (Iron Yoke - steel_1008)
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
    apply_standard_config(iron_yoke, int_pts=3000, surf_pts=1600, pde_nonlin_w=2.0)

    # 2. Nam châm trái (N30UH)
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
    apply_standard_config(magnet_left, int_pts=2000, surf_pts=1200, pde_nonlin_w=1.0)

    # 3. Nam châm phải (N30UH)
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
    apply_standard_config(magnet_right, int_pts=2000, surf_pts=1200, pde_nonlin_w=1.0)

    # 4. Background Air
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        pde_nonlinear_weight=1.0,
        interface_weight=1.0,
        pde_fading_beta=80.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=8000,
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
    
    print("[*] Chạy tiến trình huấn luyện kiểm thử...")
    model.execute_training_process()

    print("[*] Đang đánh giá và xuất biểu đồ từ trường bằng High-Level API...")
    model.plot_fields(
        resolution=180, 
        save_path=f"{output_dir}/modern_final_fields.png"
    )

    print(f"[✔] HOÀN THÀNH BÀI KIỂM THỬ TRONG 1 DÒNG LỆNH VẼ BIỂU ĐỒ! Kết quả lưu tại {output_dir}/")

def test_simulation4():
    run_modern_simulation_test()

if __name__ == "__main__":
    run_modern_simulation_test()
