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


def run_fpi_visualization_simulation_test():
    output_dir = "figure_fpi_visualization_simulation_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("[*] KHỞI TẠO MÔ PHỎNG XPINN (BẬT TÍNH NĂNG XUẤT FRAME / GIF / MP4)...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # =========================================================================
    # 1. KHUNG TỌA ĐỘ VÀ QUY MÔ VẬT LÝ TOÀN CỤC
    # =========================================================================
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 960000.0
    sim_domain.physics_config.scale_nu0 = 795774.715459
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    # =========================================================================
    # 2. CẤU HÌNH LẤY MẪU ĐỘNG (COLLOCATION SAMPLER CONFIG)
    # =========================================================================
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 50
    model.sampler_config.resample_points_before_lbfgs_stage = False

    # =========================================================================
    # 3. CẤU HÌNH TIẾN TRÌNH HUẤN LUYỆN VÀ OPTIMIZER
    # =========================================================================
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = False  # Tập trung quan sát FPI

    # Cấu hình 200 epochs Warm-up + 10 vòng FPI (mỗi vòng 100 epochs) = 1,200 Epochs
    epochs_warmup_target = 200
    fpi_iterations = 10
    
    model.training_config.fpi_outer_iterations = fpi_iterations
    model.training_config.number_of_adam_epochs = epochs_warmup_target + (fpi_iterations * 100)
    model.training_config.linear_warmup_epochs_ratio = epochs_warmup_target / model.training_config.number_of_adam_epochs

    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-6
    model.training_config.adam_eta_min = 1e-5
    model.training_config.grad_clip_max_norm = 1.0
    model.training_config.max_non_finite_streak = 5
    model.training_config.adam_log_interval = 50

    model.training_config.fpi_alpha_damping = 0.35  # Hệ số giảm chấn FPI vừa phải
    model.training_config.fpi_tolerance = 1e-6

    # Cấu hình huấn luyện Mạng phụ Net_nu
    model.training_config.pretrain_nu_epochs = 150
    model.training_config.pretrain_nu_lr = 5e-3
    model.training_config.fit_nu_epochs = 100
    model.training_config.fit_nu_lr = 2e-3

    # =========================================================================
    # 4. CẤU HÌNH TRỰC QUAN HÓA (BẬT XUẤT FRAME / GIF / MP4)
    # =========================================================================
    model.visualization_config.active = True                  # BẬT LƯU FRAME TỰ ĐỘNG
    model.visualization_config.update_interval = 50           # Cứ 50 epochs lưu 1 frame ảnh
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 120               # Độ phân giải ảnh frame (80-120 vừa đẹp và lưu nhanh)
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "training_process.mp4"
    model.visualization_config.animation_fps = 2.0            # 2 khung hình / giây
    model.visualization_config.gif_fps = 2.0
    model.visualization_config.contour_levels = 60

    # Cấu hình dải màu hiển thị mu_r
    model.visualization_config.mur_scale = "linear"
    model.visualization_config.mur_vmin = 0.0
    model.visualization_config.mur_vmax = 3000.0
    model.visualization_config.mur_cmap = "turbo"

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
    iron_yoke.training_config.interface_ht_weight = 1000.0
    iron_yoke.training_config.pde_fading_beta = 80.0
    iron_yoke.training_config.sampling_density_multiplier = 1.0
    iron_yoke.training_config.number_of_interior_points = 3000
    iron_yoke.training_config.number_of_surface_points = 1500

    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128

    iron_yoke.nu_network_config.hidden_layers = 2
    iron_yoke.nu_network_config.hidden_neurons = 32
    iron_yoke.nu_network_config.activation_function = nn.SiLU()
    iron_yoke.nu_network_config.fourier_features = False

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

    magnet_left.training_config.number_of_interior_points = 1200
    magnet_left.training_config.number_of_surface_points = 800

    magnet_left.network_config.hidden_layers = 3
    magnet_left.network_config.hidden_neurons = 48
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.0
    magnet_left.network_config.fourier_mapping_size = 64

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

    magnet_right.training_config.number_of_interior_points = 1200
    magnet_right.training_config.number_of_surface_points = 800

    magnet_right.network_config.hidden_layers = 3
    magnet_right.network_config.hidden_neurons = 48
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.0
    magnet_right.network_config.fourier_mapping_size = 64

    magnet_right.build_domain()

    # --- MIỀN 4: KHÔNG KHÍ NỀN (Background Air Domain) ---
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        interface_weight=1.0,
        interface_potential_weight=500.0,
        interface_bn_weight=800.0,
        interface_ht_weight=1000.0,
        pde_fading_beta=80.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=4000,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=3,
        hidden_neurons=48,
        activation_function=nn.SiLU(),
        fourier_features=True,
        fourier_sigma=1.0,
        fourier_mapping_size=64
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    # =========================================================================
    # 6. THỰC THI HUẤN LUYỆN VÀ TẠO ANIMATION
    # =========================================================================
    model.update_electromagnetic_pinn()

    print("[*] Bắt đầu huấn luyện và lưu frame định kỳ...")
    model.execute_training_process()

    print("[*] Xuất biểu đồ trường kết quả cuối cùng...")
    model.plot_fields(
        resolution=160,
        save_path=f"{output_dir}/fpi_final_fields.png",
        show=False,
        figsize=(12, 10)
    )
    print(f"[✔] HOÀN THÀNH! Tất cả frame ảnh (`frame_00001.png`,...), video (`training_process.mp4`) và GIF (`training_process.gif`) đã lưu tại: {output_dir}/")


if __name__ == "__main__":
    run_fpi_visualization_simulation_test()