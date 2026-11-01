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
    
    print("[*] Khởi tạo XPINN - TỐI ƯU HÓA KIẾN TRÚC THEO ĐẶC TÍNH VẬT LÝ MIỀN...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # =========================================================================
    # 1. CẤU HÌNH MIỀN KHÔNG GIAN VÀ QUY MÔ VẬT LÝ (PHYSICS CONFIG)
    # =========================================================================
    sim_domain.physics_config.scale_L0 = 0.12            # Tỷ lệ chuẩn hóa chiều dài (m)
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2  # Tỷ lệ chuẩn hóa cường độ từ trường (A/m)
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    # =========================================================================
    # 2. CẤU HÌNH BỘ LẤY MẪU ĐỘNG (COLLOCATION SAMPLER CONFIG)
    # =========================================================================
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 20          # Tái lấy mẫu mịn hơn sau mỗi 20 epochs
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # =========================================================================
    # 3. CẤU HÌNH SIÊU THAM SỐ HUẤN LUYỆN ĐỂ HẠ THẤP NGƯỠNG LOSS
    # =========================================================================
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = False
    model.training_config.enable_lbfgs_refinement_stage = True

    # --- Bộ giải Adam (Mở rộng quy mô hội tụ) ---
    model.training_config.number_of_adam_epochs = 5000            # Tăng lên 5000 epochs
    model.training_config.adam_learning_rate = 8e-4
    model.training_config.target_loss_threshold = 1e-9

    # --- Bộ giải L-BFGS (Ép sâu nghiệm) ---
    model.training_config.number_of_lbfgs_iterations = 2000       # Tăng lên 2000 iterations
    model.training_config.lbfgs_learning_rate = 0.8
    model.training_config.lbfgs_maximum_evaluations = 3000
    model.training_config.lbfgs_tolerance_gradient = 1e-14       # Siết chặt dung sai dừng lặp
    model.training_config.lbfgs_tolerance_change = 1e-14
    model.training_config.lbfgs_history_size = 100               # Mở rộng bộ nhớ Hessian

    # =========================================================================
    # 4. CẤU HÌNH TRỰC QUAN HÓA & XUẤT VIDEO
    # =========================================================================
    model.visualization_config.active = True
    model.visualization_config.update_interval = 20               # Lưu 1 frame mỗi 20 epochs
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 120                   # Độ phân giải lưới trực quan
    model.visualization_config.generate_gif_animation = False
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "deep_optimized_training.mp4"
    model.visualization_config.animation_fps = 12

    # =========================================================================
    # 5. CẤU HÌNH DUNG LƯỢNG MẠNG VÀ CÁC MIỀN VẬT LIỆU
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # Miền 1: Ách sắt (Mạng mở rộng: 5 layers x 128 neurons, Fourier Size 256)
    # -------------------------------------------------------------------------
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name="steel_1008",
        outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    )
    iron_yoke.physics_config.is_nonlinear = False
    iron_yoke.training_config.pde_weight = 1.2
    iron_yoke.training_config.interface_weight = 2.0             # Tăng trọng số giao diện
    iron_yoke.training_config.interface_potential_weight = 2000.0
    iron_yoke.training_config.interface_bn_weight = 1000.0
    iron_yoke.training_config.interface_ht_weight = 1000.0
    iron_yoke.training_config.pde_fading_beta = 120.0
    iron_yoke.training_config.number_of_interior_points = 6000    # Tăng số điểm lấy mẫu
    iron_yoke.training_config.number_of_surface_points = 3000

    iron_yoke.network_config.hidden_layers = 5                    # Tăng từ 4 lên 5 layers
    iron_yoke.network_config.hidden_neurons = 128                 # Tăng từ 64 lên 128 neurons
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.5
    iron_yoke.network_config.fourier_mapping_size = 256           # Tăng từ 128 lên 256 features
    iron_yoke.build_domain()

    # -------------------------------------------------------------------------
    # Miền 2 & 3: Nam châm trái/phải (Mạng thu gọn: 3 layers x 48 neurons)
    # -------------------------------------------------------------------------
    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left", material_name="N30UH",
        outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_weight = 2.0
    magnet_left.training_config.number_of_interior_points = 2000
    magnet_left.training_config.number_of_surface_points = 1000

    magnet_left.network_config.hidden_layers = 3                  # Thu gọn mạng
    magnet_left.network_config.hidden_neurons = 48
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.0
    magnet_left.network_config.fourier_mapping_size = 64
    magnet_left.build_domain()

    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right", material_name="N30UH",
        outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_weight = 2.0
    magnet_right.training_config.number_of_interior_points = 2000
    magnet_right.training_config.number_of_surface_points = 1000

    magnet_right.network_config.hidden_layers = 3                  # Thu gọn mạng
    magnet_right.network_config.hidden_neurons = 48
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.0
    magnet_right.network_config.fourier_mapping_size = 64
    magnet_right.build_domain()

    # -------------------------------------------------------------------------
    # Miền 4: Background Air Domain (Mạng LỚN NHẤT: 6 layers x 128 neurons, 18.000 điểm)
    # -------------------------------------------------------------------------
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.2, 
        interface_weight=2.0, 
        pde_fading_beta=120.0,
        sampling_density_multiplier=1.0, 
        number_of_interior_points=18000,                           # Tăng mạnh từ 10.000 lên 18.000 điểm
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=6,                                           # Mạng sâu nhất 6 layers
        hidden_neurons=128,                                        # 128 neurons mỗi tầng
        activation_function=nn.SiLU(),
        fourier_features=True, 
        fourier_sigma=1.5, 
        fourier_mapping_size=256                                   # Tăng Fourier Mapping lên 256
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    model.update_electromagnetic_pinn()
    
    # =========================================================================
    # 6. THỰC THI THUẬT TOÁN
    # =========================================================================
    print("[*] Bắt đầu chạy tiến trình huấn luyện tối ưu hóa sâu...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ...")
    model.plot_fields(resolution=200, save_path=f"{output_dir}/optimized_final_fields.png")
    print(f"[✔] HOÀN THÀNH KIỂM THỬ! Kết quả lưu tại: {output_dir}/")

if __name__ == "__main__":
    run_optimized_simulation_test()