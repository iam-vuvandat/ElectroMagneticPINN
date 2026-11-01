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

def run_fpi_simulation_test():
    output_dir = "figure_fpi_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu khởi tạo hệ thống mô phỏng XPINN...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # =========================================================================
    # 1. CẤU HÌNH MIỀN KHÔNG GIAN VÀ QUY MÔ VẬT LÝ (PHYSICS CONFIG)
    # =========================================================================
    sim_domain.physics_config.scale_L0 = 0.12            # Tỷ lệ chuẩn hóa chiều dài (m)
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2  # Tỷ lệ chuẩn hóa cường độ từ trường (A/m)
    sim_domain.physics_config.allow_overlapping_domains = False  # Không cho phép các miền chồng lấn
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)           # Biên không gian X (min, max)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)           # Biên không gian Y (min, max)

    # =========================================================================
    # 2. CẤU HÌNH BỘ LẤY MẪU ĐỘNG (COLLOCATION SAMPLER CONFIG)
    # =========================================================================
    model.sampler_config.dynamic_resampling = True                   # Bật tái lấy mẫu điểm sau mỗi chu kỳ
    model.sampler_config.resample_interval = 100                     # Khoảng cách số epoch tái lấy mẫu
    model.sampler_config.resample_points_before_lbfgs_stage = True  # Tái lấy mẫu lại toàn bộ trước khi chạy L-BFGS

    # =========================================================================
    # 3. CẤU HÌNH TIẾN TRÌNH HUẤN LUYỆN (TRAINING CONFIG)
    # =========================================================================
    # --- Điều khiển các giai đoạn (Stages Control) ---
    model.training_config.enable_linear_warmup_stage = True        # Giai đoạn 1A: Adam Warm-up tuyến tính
    model.training_config.enable_nonlinear_adaptation_stage = True # Giai đoạn 1B: Vòng lặp FPI phi tuyến
    model.training_config.enable_lbfgs_refinement_stage = True     # Giai đoạn 2: Tinh chỉnh bằng L-BFGS
    model.training_config.linear_warmup_epochs_ratio = 0.7        # Tỷ lệ epoch dành cho Warm-up 

    # --- Cấu hình bộ giải Adam ---
    model.training_config.number_of_adam_epochs = 5000            # Tổng số epoch cho giai đoạn Adam
    model.training_config.adam_learning_rate = 1e-3               # Tốc độ học ban đầu của Adam
    model.training_config.target_loss_threshold = 1e-5            # Ngưỡng Loss mục tiêu

    # --- Cấu hình Vòng lặp ngoại FPI (Fixed-Point Iteration) ---
    model.training_config.fpi_outer_iterations = 20               # Số vòng lặp ngoại tối đa
    model.training_config.fpi_alpha_damping = 0.1                # Hệ số giảm chấn alpha (0.0 < alpha <= 1.0)
    model.training_config.fpi_tolerance = 1e-4                   # Ngưỡng hội tụ sai số dNu để dừng FPI sớm

    # --- Cấu hình bộ giải L-BFGS ---
    model.training_config.number_of_lbfgs_iterations = 800       # Số bước lặp L-BFGS tối đa
    model.training_config.lbfgs_learning_rate = 0.8              # Tốc độ học L-BFGS
    model.training_config.lbfgs_maximum_evaluations = 1000        # Số lần đánh giá hàm closure tối đa
    model.training_config.lbfgs_tolerance_gradient = 1e-8        # Dung sai gradient dừng lặp
    model.training_config.lbfgs_tolerance_change = 1e-10         # Dung sai thay đổi nghiệm dừng lặp
    model.training_config.lbfgs_history_size = 50               # Kích thước bộ nhớ lịch sử Hessian

    # =========================================================================
    # 4. CẤU HÌNH TRỰC QUAN HÓA & XUẤT VIDEO (VISUALIZATION CONFIG)
    # =========================================================================
    model.visualization_config.active = True                       # Bật/tắt toàn bộ tiến trình lưu ảnh/video
    model.visualization_config.update_interval = 10              # Tần số lưu frame (Lưu 1 frame sau mỗi 100 epochs)
    model.visualization_config.output_directory = output_dir       # Thư mục lưu trữ hình ảnh và video
    model.visualization_config.resolution = 100                    # Độ phân giải lưới tọa độ khi vẽ trường (100x100)
    
    # --- Cấu hình xuất GIF ---
    model.visualization_config.generate_gif_animation = False      # TẮT tạo file ảnh động GIF
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.gif_fps = 5                         # Tốc độ khung hình GIF

    # --- Cấu hình xuất MP4 ---
    model.visualization_config.generate_mp4_video = True           # BẬT tạo file video MP4
    model.visualization_config.mp4_filename = "fpi_training_process.mp4" # Tên file video MP4 xuất ra
    model.visualization_config.animation_fps = 10                  # Tốc độ khung hình video MP4 (10 fps)
    model.visualization_config.mp4_codec = "libx264"               # Chuẩn nén Codec video
    model.visualization_config.mp4_quality = 8                     # Chất lượng xuất video (mức 1-10)

    # =========================================================================
    # 5. KHỞI TẠO CÁC MIỀN VẬT LIỆU VÀ CẤU HÌNH MẠNG CON (MATERIAL DOMAINS)
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # Miền 1: Ách sắt phi tuyến (Steel 1008)
    # -------------------------------------------------------------------------
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name="steel_1008",
        outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    )
    iron_yoke.physics_config.is_nonlinear = True
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_weight = 1.5
    iron_yoke.training_config.interface_potential_weight = 1000.0
    iron_yoke.training_config.interface_bn_weight = 500.0
    iron_yoke.training_config.interface_ht_weight = 500.0
    iron_yoke.training_config.pde_fading_beta = 100.0
    iron_yoke.training_config.number_of_interior_points = 4000
    iron_yoke.training_config.number_of_surface_points = 2000

    # Mạng chính Net_Az
    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128

    # Mạng phụ Net_nu (dành cho miền phi tuyến)
    iron_yoke.nu_network_config.hidden_layers = 2
    iron_yoke.nu_network_config.hidden_neurons = 32
    iron_yoke.nu_network_config.activation_function = nn.SiLU()
    iron_yoke.nu_network_config.fourier_features = False
    iron_yoke.build_domain()

    # -------------------------------------------------------------------------
    # Miền 2: Nam châm trái (N30UH - Hướng từ hóa đi lên)
    # -------------------------------------------------------------------------
    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left", material_name="N30UH",
        outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_weight = 1.5
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
    # Miền 3: Nam châm phải (N30UH - Hướng từ hóa đi xuống)
    # -------------------------------------------------------------------------
    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right", material_name="N30UH",
        outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_weight = 1.5
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
    # Miền 4: Miền không khí nền (Background Air Domain)
    # -------------------------------------------------------------------------
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0, interface_weight=1.5, pde_fading_beta=100.0,
        sampling_density_multiplier=1.0, number_of_interior_points=10000, number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4, hidden_neurons=64, activation_function=nn.SiLU(),
        fourier_features=True, fourier_sigma=1.2, fourier_mapping_size=128
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    # Cập nhật và đồng bộ toàn bộ kiến trúc mô hình
    model.update_electromagnetic_pinn()
    
    # =========================================================================
    # 6. THỰC THI THUẬT TOÁN VÀ XUẤT KẾT QUẢ
    # =========================================================================
    print("[*] Chạy tiến trình huấn luyện FPI...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ...")
    model.plot_fields(resolution=180, save_path=f"{output_dir}/fpi_final_fields.png")
    print(f"[✔] HOÀN THÀNH KIỂM THỬ FPI! Kết quả và video MP4 được lưu tại: {output_dir}/")

if __name__ == "__main__":
    run_fpi_simulation_test()