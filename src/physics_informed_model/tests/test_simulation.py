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


def run_full_explicit_config_simulation_test():
    output_dir = "figure_full_config_simulation_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("[*] KHỞI TẠO MÔ PHỎNG XPINN VỚI TOÀN BỘ CẤU HÌNH ĐƯỢC KHAI BÁO TƯỜNG MINH...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # =========================================================================
    # 1. KHUNG TỌA ĐỘ VÀ QUY MÔ VẬT LÝ TOÀN CỤC
    # =========================================================================
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 960000.0  # 800000.0 * 1.2
    sim_domain.physics_config.scale_nu0 = 795774.715459
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    # =========================================================================
    # 2. CẤU HÌNH LẤY MẪU ĐỘNG (COLLOCATION SAMPLER CONFIG)
    # =========================================================================
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 40
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # =========================================================================
    # 3. CẤU HÌNH TIẾN TRÌNH HUẤN LUYỆN VÀ OPTIMIZER (TRAINING CONFIG)
    # =========================================================================
    # Bật/Tắt các giai đoạn
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = True

    # Tiến trình Adam
    model.training_config.number_of_adam_epochs = 6000
    model.training_config.linear_warmup_epochs_ratio = 0.25
    model.training_config.adam_learning_rate = 8e-4
    model.training_config.target_loss_threshold = 1e-8
    model.training_config.adam_eta_min = 1e-6
    model.training_config.grad_clip_max_norm = 1.0
    model.training_config.max_non_finite_streak = 5
    model.training_config.adam_log_interval = 100

    # Tiến trình lặp điểm cố định phi tuyến (FPI Outer Loop)
    model.training_config.fpi_outer_iterations = 12
    model.training_config.fpi_alpha_damping = 0.5
    model.training_config.fpi_tolerance = 1e-6

    # Cấu hình huấn luyện mạng phụ Net_nu
    model.training_config.pretrain_nu_epochs = 400
    model.training_config.pretrain_nu_lr = 8e-3
    model.training_config.fit_nu_epochs = 300
    model.training_config.fit_nu_lr = 2e-3

    # Cấu hình tối ưu hóa L-BFGS
    model.training_config.lbfgs_learning_rate = 0.4
    model.training_config.number_of_lbfgs_iterations = 2000
    model.training_config.lbfgs_maximum_evaluations = 3500
    model.training_config.lbfgs_tolerance_gradient = 1e-12
    model.training_config.lbfgs_tolerance_change = 1e-12
    model.training_config.lbfgs_history_size = 100
    model.training_config.lbfgs_log_interval = 25

    # =========================================================================
    # 4. CẤU HÌNH TRỰC QUAN HÓA CHI TIẾT (VISUALIZATION CONFIG)
    # =========================================================================
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 140
    model.visualization_config.generate_gif_animation = False
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "full_explicit_config_training.mp4"
    model.visualization_config.animation_fps = 5
    model.visualization_config.gif_fps = 5
    model.visualization_config.mp4_codec = "libx264"
    model.visualization_config.mp4_quality = 8
    model.visualization_config.contour_levels = 60

    # Cấu hình hiển thị dải màu cho từng subplot
    model.visualization_config.res_scale = "log"
    model.visualization_config.res_vmin = 1e-4
    model.visualization_config.res_vmax = 1e1
    model.visualization_config.res_cmap = "Reds"

    model.visualization_config.mur_scale = "linear"
    model.visualization_config.mur_vmin = 0.0
    model.visualization_config.mur_vmax = 3000.0
    model.visualization_config.mur_cmap = "turbo"

    model.visualization_config.az_scale = "linear"
    model.visualization_config.az_vmin = -0.03
    model.visualization_config.az_vmax = 0.005
    model.visualization_config.az_cmap = "jet"

    model.visualization_config.b_mag_scale = "linear"
    model.visualization_config.b_mag_vmin = 0.0
    model.visualization_config.b_mag_vmax = 1.8
    model.visualization_config.b_mag_cmap = "jet"

    model.visualization_config.by_scale = "linear"
    model.visualization_config.by_vmin = -1.8
    model.visualization_config.by_vmax = 1.8
    model.visualization_config.by_cmap = "coolwarm"

    # =========================================================================
    # 5. KHAI BÁO CÁC MIỀN VẬT LIỆU
    # =========================================================================

    # --- MIỀN 1: LÕI SẮT PHI TUYẾN (Iron Yoke) ---
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name="steel_1008",
        outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    )
    iron_yoke.physics_config.relative_permeability = 1000.0
    iron_yoke.physics_config.coercive = [0.0, 0.0]
    iron_yoke.physics_config.current = 0.0
    iron_yoke.physics_config.is_nonlinear = True

    iron_yoke.training_config.pde_weight = 3.0
    iron_yoke.training_config.pde_nonlinear_weight = 1.0
    iron_yoke.training_config.interface_weight = 1.0
    iron_yoke.training_config.interface_potential_weight = 500.0
    iron_yoke.training_config.interface_bn_weight = 800.0
    iron_yoke.training_config.interface_ht_weight = 1200.0
    iron_yoke.training_config.pde_fading_beta = 80.0
    iron_yoke.training_config.sampling_density_multiplier = 1.0
    iron_yoke.training_config.number_of_interior_points = 5000
    iron_yoke.training_config.number_of_surface_points = 2500

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
    iron_yoke.nu_network_config.fourier_sigma = 1.0
    iron_yoke.nu_network_config.fourier_mapping_size = 32

    iron_yoke.build_domain()

    # --- MIỀN 2: NAM CHÂM VĨNH CỬU TRÁI (Magnet Left) ---
    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left",
        material_name="N30UH",
        outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.relative_permeability = 1.05
    magnet_left.physics_config.current = 0.0
    magnet_left.physics_config.is_nonlinear = False

    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.pde_nonlinear_weight = 1.0
    magnet_left.training_config.interface_weight = 1.0
    magnet_left.training_config.interface_potential_weight = 500.0
    magnet_left.training_config.interface_bn_weight = 800.0
    magnet_left.training_config.interface_ht_weight = 1200.0
    magnet_left.training_config.pde_fading_beta = 80.0
    magnet_left.training_config.sampling_density_multiplier = 1.0
    magnet_left.training_config.number_of_interior_points = 2500
    magnet_left.training_config.number_of_surface_points = 1800

    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128

    magnet_left.nu_network_config.hidden_layers = 2
    magnet_left.nu_network_config.hidden_neurons = 32
    magnet_left.nu_network_config.activation_function = nn.SiLU()
    magnet_left.nu_network_config.fourier_features = False
    magnet_left.nu_network_config.fourier_sigma = 1.0
    magnet_left.nu_network_config.fourier_mapping_size = 32

    magnet_left.build_domain()

    # --- MIỀN 3: NAM CHÂM VĨNH CỬU PHẢI (Magnet Right) ---
    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right",
        material_name="N30UH",
        outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.relative_permeability = 1.05
    magnet_right.physics_config.current = 0.0
    magnet_right.physics_config.is_nonlinear = False

    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.pde_nonlinear_weight = 1.0
    magnet_right.training_config.interface_weight = 1.0
    magnet_right.training_config.interface_potential_weight = 500.0
    magnet_right.training_config.interface_bn_weight = 800.0
    magnet_right.training_config.interface_ht_weight = 1200.0
    magnet_right.training_config.pde_fading_beta = 80.0
    magnet_right.training_config.sampling_density_multiplier = 1.0
    magnet_right.training_config.number_of_interior_points = 2500
    magnet_right.training_config.number_of_surface_points = 1800

    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128

    magnet_right.nu_network_config.hidden_layers = 2
    magnet_right.nu_network_config.hidden_neurons = 32
    magnet_right.nu_network_config.activation_function = nn.SiLU()
    magnet_right.nu_network_config.fourier_features = False
    magnet_right.nu_network_config.fourier_sigma = 1.0
    magnet_right.nu_network_config.fourier_mapping_size = 32

    magnet_right.build_domain()

    # --- MIỀN 4: KHÔNG KHÍ NỀN (Background Air Domain) ---
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        pde_nonlinear_weight=1.0,
        interface_weight=1.0,
        interface_potential_weight=500.0,
        interface_bn_weight=800.0,
        interface_ht_weight=1200.0,
        pde_fading_beta=80.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=9000,
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

    # =========================================================================
    # 6. KHỞI TẠO VÀ THỰC THI HUẤN LUYỆN
    # =========================================================================
    model.update_electromagnetic_pinn()

    print("[*] Bắt đầu tiến trình huấn luyện với toàn bộ cấu hình cụ thể...")
    model.execute_training_process()

    print("[*] Xuất biểu đồ trường từ kết quả...")
    model.plot_fields(
        resolution=220,
        save_path=f"{output_dir}/full_explicit_config_final_fields.png",
        show=False,
        figsize=(14, 11)
    )
    print(f"[✔] HOÀN THÀNH MÔ PHỎNG! Kết quả lưu tại: {output_dir}/")


if __name__ == "__main__":
    run_full_explicit_config_simulation_test()