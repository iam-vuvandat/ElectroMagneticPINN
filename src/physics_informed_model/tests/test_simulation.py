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

def run_optimized_simulation_test():
    output_dir = "figure_linear_optimized_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] KHỞI TẠO XPINN - ĐẶT TRỌNG SỐ VẬT LÝ VÀ CỐ ĐỊNH TẬP ĐIỂM CHO L-BFGS...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. CẤU HÌNH KHUNG TOẠ ĐỘ VÀ QUY MÔ VẬT LÝ
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    # 2. CẤU HÌNH BỘ LẤY MẪU
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 50
    model.sampler_config.resample_points_before_lbfgs_stage = False

    # 3. SIÊU THAM SỐ HUẤN LUYỆN
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = False
    model.training_config.enable_lbfgs_refinement_stage = True

    model.training_config.number_of_adam_epochs = 4000
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-8

    model.training_config.number_of_lbfgs_iterations = 1500
    model.training_config.lbfgs_learning_rate = 0.5
    model.training_config.lbfgs_maximum_evaluations = 2500
    model.training_config.lbfgs_tolerance_gradient = 1e-12
    model.training_config.lbfgs_tolerance_change = 1e-12
    model.training_config.lbfgs_history_size = 50

    # 4. TRỰC QUAN HÓA
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 120
    model.visualization_config.generate_gif_animation = False
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "deep_optimized_training.mp4"

    # 5. CẤU HÌNH CÁC MIỀN VẬT LIỆU
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke", material_name="steel_1008",
        outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    )
    iron_yoke.physics_config.is_nonlinear = False
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_potential_weight = 100.0
    iron_yoke.training_config.interface_bn_weight = 50.0
    iron_yoke.training_config.interface_ht_weight = 50.0
    iron_yoke.training_config.pde_fading_beta = 60.0
    iron_yoke.training_config.number_of_interior_points = 4000
    iron_yoke.training_config.number_of_surface_points = 1500

    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128
    iron_yoke.build_domain()

    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left", material_name="N30UH",
        outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_potential_weight = 100.0
    magnet_left.training_config.interface_bn_weight = 50.0
    magnet_left.training_config.interface_ht_weight = 50.0
    magnet_left.training_config.number_of_interior_points = 2000
    magnet_left.training_config.number_of_surface_points = 1000

    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128
    magnet_left.build_domain()

    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right", material_name="N30UH",
        outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_potential_weight = 100.0
    magnet_right.training_config.interface_bn_weight = 50.0
    magnet_right.training_config.interface_ht_weight = 50.0
    magnet_right.training_config.number_of_interior_points = 2000
    magnet_right.training_config.number_of_surface_points = 1000

    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128
    magnet_right.build_domain()

    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0, 
        pde_fading_beta=60.0,
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
    
    print("[*] Chạy tiến trình huấn luyện tối ưu hóa...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ...")
    model.plot_fields(resolution=200, save_path=f"{output_dir}/optimized_final_fields.png")
    print(f"[✔] HOÀN THÀNH KIỂM THỬ! Kết quả lưu tại: {output_dir}/")

if __name__ == "__main__":
    run_optimized_simulation_test()
