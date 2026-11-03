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


def run_high_precision_simulation_test():
    output_dir = "figure_high_precision_simulation_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("[*] KHỞI TẠO MÔ PHỎNG ĐỘ CHÍNH XÁC CAO (HIGH PRECISION PHYSICAL SIMULATION)...")

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
    model.sampler_config.dynamic_resampling = True           # Bật resampling để tập trung điểm vào vùng sai số lớn
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # =========================================================================
    # 3. CẤU HÌNH TIẾN TRÌNH HUẤN LUYỆN CHÍNH XÁC CAO
    # =========================================================================
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = True    # Bật L-BFGS tinh chỉnh tối ưu nghiệm cuối

    epochs_warmup_target = 800
    fpi_iterations = 12                                        # 12 vòng lặp FPI cho hội tụ phi tuyến bão hòa
    epochs_per_fpi = 400                                       # 400 Adam epochs cho mỗi vòng FPI
    
    model.training_config.fpi_outer_iterations = fpi_iterations
    model.training_config.fpi_alpha_damping = 0.7             # Damping 0.7 chống dao động nghiệm từ trở
    model.training_config.fpi_tolerance = 1e-6

    model.training_config.number_of_adam_epochs = epochs_warmup_target + (fpi_iterations * epochs_per_fpi)
    model.training_config.linear_warmup_epochs_ratio = epochs_warmup_target / model.training_config.number_of_adam_epochs

    model.training_config.adam_learning_rate = 1e-3
    model.training_config.adam_log_interval = 50

    # Cấu hình mạng Net_nu (Mô hình hóa chính xác độ từ trở phi tuyến)
    model.training_config.pretrain_nu_epochs = 400             # Pre-train kỹ cho Net_nu ban đầu
    model.training_config.pretrain_nu_lr = 5e-3
    model.training_config.fit_nu_epochs = 250                  # Fit nu kỹ ở từng vòng FPI
    model.training_config.fit_nu_lr = 2e-3

    model.training_config.number_of_lbfgs_iterations = 800     # Tinh chỉnh L-BFGS sâu

    # =========================================================================
    # 4. CẤU HÌNH TRỰC QUAN HÓA (THANG LOGARITHMIC DÙNG CHO MUR ĐẢM BẢO CHÍNH XÁC)
    # =========================================================================
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100           # Cứ 100 epochs xuất 1 frame
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 160                # Độ phân giải cao cho ảnh công bố
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "high_precision_mur_training_process.gif"
    model.visualization_config.generate_mp4_video = False
    model.visualization_config.animation_fps = 5.0
    model.visualization_config.gif_fps = 5.0
    model.visualization_config.contour_levels = 80

    # Thang Logarithmic cho mur để bắt trọn dải mur từ 8.3 đến ~7000
    model.visualization_config.mur_scale = "log"
    model.visualization_config.mur_vmin = 1.0
    model.visualization_config.mur_vmax = 7000.0               
    model.visualization_config.mur_cmap = "turbo"

    # =========================================================================
    # 5. KHAI BÁO CÁC MIỀN VẬT LIỆU (MẬT ĐỘ ĐIỂM CAO & MẠNG MỞ RỘNG DUNG LƯỢNG)
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

    iron_yoke.training_config.pde_weight = 4.0
    iron_yoke.training_config.interface_weight = 5.0           # Tăng trọng số liên tục biên
    iron_yoke.training_config.number_of_interior_points = 4000  # Mật độ điểm cao
    iron_yoke.training_config.number_of_surface_points = 1500

    iron_yoke.network_config.hidden_layers = 4                 # Mạng mở rộng dung lượng
    iron_yoke.network_config.hidden_neurons = 96
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.5
    iron_yoke.network_config.fourier_mapping_size = 96

    iron_yoke.nu_network_config.hidden_layers = 3
    iron_yoke.nu_network_config.hidden_neurons = 48
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

    magnet_left.training_config.number_of_interior_points = 1800
    magnet_left.training_config.number_of_surface_points = 800

    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
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

    magnet_right.training_config.number_of_interior_points = 1800
    magnet_right.training_config.number_of_surface_points = 800

    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 64

    magnet_right.build_domain()

    # --- MIỀN 4: KHÔNG KHÍ NỀN (Background Air Domain) ---
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.5,
        interface_weight=3.0,
        number_of_interior_points=7500,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=64,
        activation_function=nn.SiLU(),
        fourier_features=True,
        fourier_sigma=1.2,
        fourier_mapping_size=64
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    # =========================================================================
    # 6. THỰC THI HUẤN LUYỆN ĐỘ CHÍNH XÁC CAO VÀ XUẤT KẾT QUẢ
    # =========================================================================
    model.update_electromagnetic_pinn()

    print("[*] Bắt đầu tiến trình huấn luyện chính xác cao...")
    model.execute_training_process()

    print("[*] Xuất biểu đồ trường kết quả cuối cùng...")
    model.plot_fields(
        resolution=180,
        save_path=f"{output_dir}/high_precision_final_fields.png",
        show=False,
        figsize=(14, 11)
    )
    print(f"[✔] HOÀN THÀNH MÔ PHỎNG VẬT LÝ CHÍNH XÁC CAO! Kết quả xem tại: {output_dir}/")


if __name__ == "__main__":
    run_high_precision_simulation_test()