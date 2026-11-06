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

output_dir = "figure_high_precision_simulation_test"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

model = ElectroMagneticPINN()
sim_domain = model.simulation_domain_engine_instance

# =========================================================================
# 1. QUY MÔ VẬT LÝ VÀ KHUNG TỌA ĐỘ (MET)
# =========================================================================
sim_domain.physics_config.scale_L0 = 0.12            # Scale chiều dài 120 mm
sim_domain.physics_config.scale_H0 = 960000.0         # Scale cường độ từ trường 960 kA/m
sim_domain.physics_config.scale_nu0 = 795774.715459   # Độ từ kháng chân không nu0
sim_domain.physics_config.allow_overlapping_domains = False
sim_domain.x_boundaries_tuple = (-0.12, 0.12)         # Khung biên X: [-120mm, +120mm]
sim_domain.y_boundaries_tuple = (-0.12, 0.12)         # Khung biên Y: [-120mm, +120mm]

# =========================================================================
# 2. CẤU HÌNH TIẾN TRÌNH HUẤN LUYỆN & VÒNG LẶP FPI
# =========================================================================
model.sampler_config.dynamic_resampling = True
model.sampler_config.resample_interval = 100
model.sampler_config.resample_points_before_lbfgs_stage = True

model.training_config.enable_linear_warmup_stage = True
model.training_config.enable_nonlinear_adaptation_stage = True
model.training_config.enable_lbfgs_refinement_stage = True

epochs_warmup_target = 1000
fpi_iterations = 5
epochs_per_fpi = 1000

model.training_config.fpi_outer_iterations = fpi_iterations
model.training_config.fpi_alpha_damping = 0.5            # Damping mượt cho FPI
model.training_config.fpi_tolerance = 1e-6

model.training_config.number_of_adam_epochs = epochs_warmup_target + (fpi_iterations * epochs_per_fpi)
model.training_config.linear_warmup_epochs_ratio = epochs_warmup_target / model.training_config.number_of_adam_epochs

model.training_config.adam_learning_rate = 2e-3
model.training_config.target_loss_threshold = 1e-7
model.training_config.adam_eta_min = 1e-6
model.training_config.grad_clip_max_norm = 1.0
model.training_config.max_non_finite_streak = 5
model.training_config.adam_log_interval = 100

model.training_config.pretrain_nu_epochs = 600
model.training_config.pretrain_nu_lr = 5e-3
model.training_config.fit_nu_epochs = 500
model.training_config.fit_nu_lr = 3e-3

model.training_config.number_of_lbfgs_iterations = 1500
model.training_config.lbfgs_learning_rate = 0.5

# =========================================================================
# 3. CẤU HÌNH TRỰC QUAN HÓA ĐẦY ĐỦ (VISUALIZATION CONFIG)
# =========================================================================
vis_cfg = model.visualization_config
vis_cfg.active = True
vis_cfg.display_colab_frames = True
vis_cfg.update_interval = 10
vis_cfg.output_directory = output_dir
vis_cfg.resolution = 160

# --- TÙY CHỌN XUẤT ANIMATION ---
vis_cfg.generate_gif_animation = True
vis_cfg.gif_filename = "ansys_fem_training_process.gif"
vis_cfg.generate_mp4_video = True
vis_cfg.mp4_filename = "ansys_fem_training_process.mp4"
vis_cfg.animation_fps = 10.0
vis_cfg.gif_fps = 10.0
vis_cfg.mp4_codec = "libx264"
vis_cfg.mp4_quality = 8
vis_cfg.contour_levels = 80

# --- A. CẤU HÌNH BIỂU ĐỒ TRƯỜNG TỪ THÔNG |B| (ANSYS FEM COLORMAP) ---
vis_cfg.b_mag_scale = "linear"
vis_cfg.b_mag_cmap = "ansys_fem"          # Mã màu chuẩn Ansys FEM (hoặc 'jet', 'turbo', 'viridis')
vis_cfg.b_mag_unit = "mTesla"             # Đơn vị hiển thị: "mTesla" hoặc "Tesla"
vis_cfg.b_mag_vmin = 0.0                 # Giá trị cực tiểu (0 mT)
vis_cfg.b_mag_vmax = 1800.0              # Giá trị cực đại (1800 mT = 1.8 T)
vis_cfg.b_mag_number_of_division = 100   # Số dải màu nội suy mượt (None hoặc 20 cho 20 màu gốc)

# --- B. CẤU HÌNH BIỂU ĐỒ SAI SỐ PDE RESIDUAL ERROR (DISCRETE LOG STEPS) ---
vis_cfg.res_scale = "log"
vis_cfg.res_vmin = 1e-4
vis_cfg.res_vmax = 10.0
vis_cfg.res_cmap = "inferno"              # Bảng màu tương phản cao ('inferno', 'magma', 'Reds')
vis_cfg.res_discrete_levels = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]  # Các mốc dải chia loga đồng mức

# --- C. CẤU HÌNH BIỂU ĐỒ ĐỘ TỪ THẤM TƯƠNG ĐỐI \mu_r ---
vis_cfg.mur_scale = "linear"
vis_cfg.mur_vmin = 0.0
vis_cfg.mur_vmax = 3000.0
vis_cfg.mur_cmap = "turbo"

# --- D. CẤU HÌNH BIỂU ĐỒ THẾ VECTƠ TỪ A_z ---
vis_cfg.az_scale = "linear"
vis_cfg.az_vmin = None                   # Tự động lấy Min
vis_cfg.az_vmax = None                   # Tự động lấy Max
vis_cfg.az_cmap = "jet"

# --- E. CẤU HÌNH BIỂU ĐỒ THÀNH PHẦN B_y ---
vis_cfg.by_scale = "linear"
vis_cfg.by_vmin = None
vis_cfg.by_vmax = None
vis_cfg.by_cmap = "coolwarm"

# =========================================================================
# 4. KHAI BÁO CÁC MIỀN HÌNH HỌC TỪ ANSYS MAXWELL
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

iron_yoke.training_config.pde_weight = 8.0                   # Trọng số PDE nội miền
iron_yoke.training_config.pde_nonlinear_weight = 2.0
iron_yoke.training_config.interface_weight = 1.0
iron_yoke.training_config.interface_potential_weight = 800.0
iron_yoke.training_config.interface_bn_weight = 1000.0
iron_yoke.training_config.interface_ht_weight = 1200.0
iron_yoke.training_config.pde_fading_beta = 80.0
iron_yoke.training_config.sampling_density_multiplier = 1.0

iron_yoke.training_config.number_of_interior_points = 15000
iron_yoke.training_config.number_of_surface_points = 4000

iron_yoke.network_config.hidden_layers = 4
iron_yoke.network_config.hidden_neurons = 128
iron_yoke.network_config.activation_function = nn.Mish()
iron_yoke.network_config.fourier_features = True
iron_yoke.network_config.fourier_sigma = 1.3
iron_yoke.network_config.fourier_mapping_size = 128

iron_yoke.nu_network_config.hidden_layers = 3
iron_yoke.nu_network_config.hidden_neurons = 64
iron_yoke.nu_network_config.activation_function = nn.Mish()
iron_yoke.nu_network_config.fourier_features = True
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

magnet_left.training_config.number_of_interior_points = 1500
magnet_left.training_config.number_of_surface_points = 800

magnet_left.network_config.hidden_layers = 4
magnet_left.network_config.hidden_neurons = 64
magnet_left.network_config.activation_function = nn.Mish()
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

magnet_right.training_config.number_of_interior_points = 1500
magnet_right.training_config.number_of_surface_points = 800

magnet_right.network_config.hidden_layers = 4
magnet_right.network_config.hidden_neurons = 64
magnet_right.network_config.activation_function = nn.Mish()
magnet_right.network_config.fourier_features = True
magnet_right.network_config.fourier_sigma = 1.0
magnet_right.network_config.fourier_mapping_size = 64

magnet_right.build_domain()

# --- MIỀN 4: KHÔNG KHÍ NỀN (Background Air Domain) ---
bg_training_cfg = SimpleNamespace(
    pde_weight=1.0,
    interface_weight=1.0,
    interface_potential_weight=800.0,
    interface_bn_weight=1000.0,
    interface_ht_weight=1200.0,
    pde_fading_beta=80.0,
    sampling_density_multiplier=1.0,
    number_of_interior_points=6000,
    number_of_surface_points=0
)
bg_network_cfg = SimpleNamespace(
    hidden_layers=4,
    hidden_neurons=64,
    activation_function=nn.Mish(),
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
# 5. CẬP NHẬT MÔ HÌNH VÀ THỰC THI HUẤN LUYỆN
# =========================================================================
model.update_electromagnetic_pinn()

print("[*] Bắt đầu huấn luyện mô hình PINN với cấu hình Ansys FEM Colormap & Discrete Log Residual...")
model.execute_training_process()

print("[*] Xuất biểu đồ trường kết quả cuối cùng...")
model.plot_fields(
    resolution=200,
    save_path=f"{output_dir}/fpi_final_fields.png",
    show=False,
    figsize=(12, 10)
)
print(f"[✔] HOÀN THÀNH! Kết quả đã được lưu tại: {output_dir}/")