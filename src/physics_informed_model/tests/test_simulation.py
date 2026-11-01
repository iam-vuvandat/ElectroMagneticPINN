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

def run_nonlinear_simulation_test():
    output_dir = "figure_nonlinear_optimized_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] KHỞI TẠO XPINN PHI TUYẾN - DAMPING = 1.0 & TRỌNG SỐ GIAO DIỆN NÂNG CAO...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. CẤU HÌNH KHUNG TOẠ ĐỘ VÀ QUY MÔ VẬT LÝ
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    # 2. CẤU HÌNH LẤY MẪU
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 50
    model.sampler_config.resample_points_before_lbfgs_stage = False

    # 3. TOÀN BỘ CẤU HÌNH THAM SỐ HUẤN LUYỆN
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = True

    # Tổng Epochs & Tốc độ học Adam
    model.training_config.number_of_adam_epochs = 5000
    model.training_config.linear_warmup_epochs_ratio = 0.30       # ~1500 Epochs Warm-up
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-8

    # Cấu hình FPI Outer Loop
    model.training_config.fpi_outer_iterations = 10               # 10 vòng lặp FPI ngoại
    model.training_config.fpi_alpha_damping = 1.0                # Cập nhật Damping factor = 1.0 (Full Update)
    model.training_config.fpi_tolerance = 1e-5                    # Dung sai FPI

    # Cấu hình Khớp Mạng phụ Net_nu
    model.training_config.pretrain_nu_epochs = 300               # Epochs Pre-train Net_nu ban đầu
    model.training_config.pretrain_nu_lr = 1e-2                  # LR Pre-train Net_nu
    model.training_config.fit_nu_epochs = 250                    # Epochs khớp Net_nu mỗi bước FPI
    model.training_config.fit_nu_lr = 3e-3                       # LR khớp Net_nu mỗi bước FPI

    # Kiểm soát Gradient & Nhật ký
    model.training_config.grad_clip_max_norm = 1.0              # Clip gradient max norm
    model.training_config.adam_eta_min = 1e-6                    # LR tối thiểu của Cosine Annealing
    model.training_config.max_non_finite_streak = 5              # Giới hạn NaN/Inf streak
    model.training_config.adam_log_interval = 100               # Tần suất in log Adam (epochs)
    model.training_config.lbfgs_log_interval = 25               # Tần suất in log L-BFGS (closures)

    # Cấu hình L-BFGS Refinement
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
    model.visualization_config.mp4_filename = "nonlinear_optimized_training.mp4"

    # 5. CẤU HÌNH CÁC MIỀN VẬT LIỆU
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke", material_name="steel_1008",
        outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    )
    iron_yoke.physics_config.is_nonlinear = True
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_potential_weight = 300.0
    iron_yoke.training_config.interface_bn_weight = 200.0         # Tăng Bn weight lên 200.0
    iron_yoke.training_config.interface_ht_weight = 500.0         # Tăng Ht weight lên 500.0
    iron_yoke.training_config.pde_fading_beta = 60.0
    iron_yoke.training_config.number_of_interior_points = 4500
    iron_yoke.training_config.number_of_surface_points = 1500

    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128

    iron_yoke.nu_network_config.hidden_layers = 3
    iron_yoke.nu_network_config.hidden_neurons = 48
    iron_yoke.nu_network_config.activation_function = nn.SiLU()
    iron_yoke.nu_network_config.fourier_features = False

    iron_yoke.build_domain()

    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left", material_name="N30UH",
        outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_potential_weight = 300.0
    magnet_left.training_config.interface_bn_weight = 200.0         # Tăng Bn weight lên 200.0
    magnet_left.training_config.interface_ht_weight = 500.0         # Tăng Ht weight lên 500.0
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
    magnet_right.training_config.interface_potential_weight = 300.0
    magnet_right.training_config.interface_bn_weight = 200.0         # Tăng Bn weight lên 200.0
    magnet_right.training_config.interface_ht_weight = 500.0         # Tăng Ht weight lên 500.0
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
    
    print("[*] Chạy tiến trình huấn luyện PHI TUYẾN...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ phi tuyến...")
    model.plot_fields(resolution=200, save_path=f"{output_dir}/nonlinear_final_fields.png")
    print(f"[✔] HOÀN THÀNH KIỂM THỬ PHI TUYẾN! Kết quả lưu tại: {output_dir}/")

if __name__ == "__main__":
    run_nonlinear_simulation_test()