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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from electro_magnetic_pinn import ElectroMagneticPINN
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY


def run_modern_simulation_test():
    output_dir = "figure_modern_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN Mô hình Hiện đại (Modern API Non-linear Test)...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # Cấu hình chuẩn hóa trường vật lý
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    # Cấu hình lấy mẫu động
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_before_lbfgs = True

    # Cấu hình số vòng huấn luyện
    model.training_config.epochs_adam = 3000
    model.training_config.epochs_lbfgs = 1500
    model.training_config.learning_rate_adam = 1e-3
    model.training_config.target_loss = 1e-7

    # Cấu hình hình ảnh hóa
    model.visualization_config.active = True
    model.visualization_config.update_interval = 50
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

    # 1. Khung Ách sắt Phi tuyến (Iron Yoke - steel_1008)
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

    # 2. Nam châm trái (N30UH - Tuyến tính)
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

    # 3. Nam châm phải (N30UH - Tuyến tính)
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
    
    print("[*] Tiến trình huấn luyện 3 Giai đoạn (Adam Linear -> Adam Non-linear -> L-BFGS)...")
    model.execute_training_process()

    print("[*] Đang đánh giá trường từ trường dự đoán...")
    resolution = 180
    xb, yb = sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    # Đưa tensor đánh giá về đúng device tính toán
    device = model.computation_device
    eval_pts = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32, device=device)
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(eval_pts)
    
    assert torch.all(torch.isfinite(A_z_pred)) and torch.all(torch.isfinite(B_x_pred)), "Lỗi: Trường dự đoán chứa NaN/Inf!"

    A_z_grid = A_z_pred.cpu().numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.cpu().numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.cpu().numpy().reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)

    # Đánh giá độ từ thẩm mu_r phân loại chính xác theo từng miền vật liệu
    props = sim_domain.evaluate_global_physical_properties(eval_pts)
    mat_ids = props["material_classification"].squeeze()
    
    B_mag_phys_tensor = torch.tensor(B_mag_grid.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    
    # Tính mu_r cho sắt
    nu_iron = iron_yoke.evaluate_reluctivity(eval_pts, B_magnitude_tensor=B_mag_phys_tensor, is_nonlinear=True)
    mu_r_iron = (VACUUM_RELUCTIVITY / torch.clamp(nu_iron, min=1e-6)).squeeze()

    # Tạo mảng mu_r tổng thể cho toàn miền
    mu_r_tensor = torch.ones_like(mat_ids, device=device) # Mặc định không khí mu_r = 1.0
    
    # Miền sắt (Iron Yoke - ID 1)
    iron_mask = (mat_ids == float(iron_yoke.domain_id))
    mu_r_tensor[iron_mask] = mu_r_iron[iron_mask]
    
    # Miền nam châm (Magnet Left & Right - ID 2, 3)
    mag_left_mask = (mat_ids == float(magnet_left.domain_id))
    mag_right_mask = (mat_ids == float(magnet_right.domain_id))
    mu_r_tensor[mag_left_mask] = getattr(magnet_left.physics_config, 'relative_permeability', 1.05)
    mu_r_tensor[mag_right_mask] = getattr(magnet_right.physics_config, 'relative_permeability', 1.05)

    mu_r_grid = mu_r_tensor.cpu().numpy().reshape(resolution, resolution)

    # Vẽ kết quả
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

    c2 = axs[1, 0].contourf(X_grid, Y_grid, mu_r_grid, levels=60, cmap="cividis")
    fig.colorbar(c2, ax=axs[1, 0], label=r"$\mu_r$")
    axs[1, 0].set_title(r"Physical Permeability Distribution ($\mu_r$)")
    axs[1, 0].set_aspect('equal')

    c3 = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig.colorbar(c3, ax=axs[1, 1], label=r"$B_y$ (T)")
    axs[1, 1].set_title(r"Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/modern_final_fields.png", dpi=150)
    plt.close('all')

    print(f"[✔] HOÀN THÀNH BÀI KIỂM THỬ MÔ HÌNH PHI TUYẾN 3 GIAI ĐOẠN! Kết quả đã được lưu tại {output_dir}/")

def test_simulation4():
    run_modern_simulation_test()

# Gọi trực tiếp ở top-level để tương thích với extract_project
run_modern_simulation_test()