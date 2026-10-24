import os
import sys
import shutil
from types import SimpleNamespace

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Thêm thư mục gốc vào sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from electro_magnetic_pinn import ElectroMagneticPINN


def run_modern_simulation_test():
    output_dir = "figure_modern_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN Mô hình Hiện đại (Modern API Test)...")

    # 1. Khởi tạo Mô hình PINN & Simulation Domain
    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # Cấu hình tỷ lệ vật lý & khung biên bài toán
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    # 2. Cấu hình Solver & Visualizer
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_before_lbfgs = True

    model.training_config.epochs_adam = 3000
    model.training_config.epochs_lbfgs = 1500
    model.training_config.learning_rate_adam = 1e-3
    model.training_config.target_loss = 1e-7

    model.visualization_config.active = True
    model.visualization_config.update_interval = 50
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "modern_training.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "modern_training.mp4"

    # Helper lambda cài đặt cấu hình mạng & huấn luyện nhanh cho các domain
    def apply_standard_config(domain, int_pts=2000, surf_pts=1200):
        domain.training_config.pde_weight = 1.0
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

    # 3. Tạo các Miền Vật liệu trực tiếp từ sim_domain (Hiện đại & Gọn gàng)
    
    # 3.1 Khung Ách sắt (Iron Yoke)
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
    apply_standard_config(iron_yoke, int_pts=3000, surf_pts=1600)

    # 3.2 Nam châm trái (Hướng từ trường +Y)
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
    apply_standard_config(magnet_left)

    # 3.3 Nam châm phải (Hướng từ trường -Y)
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
    apply_standard_config(magnet_right)

    # 3.4 Khởi tạo Miền Không khí Nền (Background Air)
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
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

    # 4. Cập nhật hệ thống & Tiến hành Huấn luyện
    model.update_electromagnetic_pinn()
    sim_domain.plot_problem_definition(resolution=100)
    
    print("[*] Tiến trình huấn luyện Adam & L-BFGS...")
    model.execute_training_process()

    # 5. Đánh giá & Xuất đồ thị kết quả
    print("[*] Đang đánh giá trường từ trường dự đoán...")
    resolution = 180
    xb, yb = sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    eval_pts = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(eval_pts)
    
    # Kiểm tra tính hợp lệ của dữ liệu
    assert torch.all(torch.isfinite(A_z_pred)) and torch.all(torch.isfinite(B_x_pred)), "Lỗi: Trường dự đoán chứa NaN/Inf!"

    A_z_grid = A_z_pred.cpu().numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.cpu().numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.cpu().numpy().reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)

    # Vẽ biểu đồ 4 ô
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    c0 = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=60, cmap="jet")
    fig.colorbar(c0, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
    axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
    axs[0, 0].set_aspect('equal')

    b_max = max(float(np.max(B_mag_grid)), 1e-3)
    c1 = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=np.linspace(0, b_max, 60), cmap="jet")
    fig.colorbar(c1, ax=axs[0, 1], label=r"$|B|$ (T)")
    axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_aspect('equal')

    c2 = axs[1, 0].contourf(X_grid, Y_grid, B_x_grid, levels=60, cmap="coolwarm")
    fig.colorbar(c2, ax=axs[1, 0], label=r"$B_x$ (T)")
    axs[1, 0].set_title(r"Field Component ($B_x$)")
    axs[1, 0].set_aspect('equal')

    c3 = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig.colorbar(c3, ax=axs[1, 1], label=r"$B_y$ (T)")
    axs[1, 1].set_title(r"Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/modern_final_fields.png", dpi=150)
    plt.close('all')

    print(f"[✔] HOÀN THÀNH KIỂM THỬ HỆ THỐNG MỚI! Kết quả đã được lưu tại {output_dir}/")


if __name__ == "__main__":
    run_modern_simulation_test()