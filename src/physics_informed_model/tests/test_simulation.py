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

sim_domain.physics_config.scale_L0 = 0.12
sim_domain.physics_config.scale_H0 = 960000.0
sim_domain.physics_config.scale_nu0 = 795774.715459
sim_domain.physics_config.allow_overlapping_domains = False
sim_domain.x_boundaries_tuple = (-0.12, 0.12)
sim_domain.y_boundaries_tuple = (-0.12, 0.12)

model.sampler_config.dynamic_resampling = True
model.sampler_config.resample_interval = 100
model.sampler_config.resample_points_before_lbfgs_stage = True


model.training_config.enable_linear_warmup_stage = True
model.training_config.enable_nonlinear_adaptation_stage = True
model.training_config.enable_lbfgs_refinement_stage = True

epochs_warmup_target = 3000
fpi_iterations = 3                        # Giảm số bước FPI từ 10-20 xuống 5
epochs_per_fpi = 3000                      # Giảm số Adam epochs mỗi bước FPI xuống 600

model.training_config.fpi_outer_iterations = fpi_iterations
model.training_config.fpi_alpha_damping = 0.8
model.training_config.fpi_tolerance = 1e-6


model.training_config.number_of_adam_epochs = epochs_warmup_target + (fpi_iterations * epochs_per_fpi)
model.training_config.linear_warmup_epochs_ratio = epochs_warmup_target / model.training_config.number_of_adam_epochs


model.training_config.adam_learning_rate = 0.2*  1e-3
model.training_config.target_loss_threshold = 1e-7
model.training_config.adam_eta_min = 1e-6
model.training_config.grad_clip_max_norm = 1.0
model.training_config.max_non_finite_streak = 5
model.training_config.adam_log_interval = 1


model.training_config.pretrain_nu_epochs = 3000   
model.training_config.pretrain_nu_lr = 5e-3
model.training_config.fit_nu_epochs = 3000        
model.training_config.fit_nu_lr = 3e-3


model.training_config.number_of_lbfgs_iterations = 3000  
model.training_config.lbfgs_learning_rate = 0.5

model.visualization_config.active = True
model.visualization_config.update_interval = 100
model.visualization_config.output_directory = output_dir
model.visualization_config.resolution = 160
model.visualization_config.generate_gif_animation = True
model.visualization_config.gif_filename = "high_precision_mur_training_process.gif"
model.visualization_config.generate_mp4_video = True
model.visualization_config.mp4_filename = "high_precision_mur_training_process.mp4"
model.visualization_config.animation_fps = 1.0
model.visualization_config.gif_fps = 1.0
model.visualization_config.mp4_codec = "libx264"
model.visualization_config.mp4_quality = 8
model.visualization_config.contour_levels = 80

# =========================================================================
# 5. CẤU HÌNH MIỀN VẬT LIỆU: TĂNG MẠNH ĐIỂM LÕI SẮT & GIẢM KÍCH THƯỚC MẠNG
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

# Trọng số cân bằng tự nhiên
iron_yoke.training_config.pde_weight = 1.0
iron_yoke.training_config.pde_nonlinear_weight = 1.0
iron_yoke.training_config.interface_weight = 1.0
iron_yoke.training_config.interface_potential_weight = 1.0
iron_yoke.training_config.interface_bn_weight = 1.0
iron_yoke.training_config.interface_ht_weight = 1.0
iron_yoke.training_config.pde_fading_beta = 80.0
iron_yoke.training_config.sampling_density_multiplier = 2.5

# TĂNG MẠNH SỐ ĐIỂM LẤY MẪU TRONG LÕI SẮT ĐỂ GIẢM MẠNH LOSS NỘI MIỀN
iron_yoke.training_config.number_of_interior_points = 30000  # Tăng mạnh từ 8,000 -> 25,000 điểm
iron_yoke.training_config.number_of_surface_points = 6000    # Tăng từ 4,000 -> 6,000 điểm biên

# GIẢM KÍCH THƯỚC MẠNG NƠ-RON (Gọn nhẹ hơn để tăng tốc độ tính toán)
iron_yoke.network_config.hidden_layers = 4                    # Giảm từ 5 -> 3 hidden layers
iron_yoke.network_config.hidden_neurons = 128                  # Giảm từ 128 -> 64 neurons
iron_yoke.network_config.activation_function = nn.Mish()
iron_yoke.network_config.fourier_features = False
iron_yoke.network_config.fourier_sigma = 1.5                  # Tăng sigma để giải quyết biến thiên độ dốc
iron_yoke.network_config.fourier_mapping_size = 128           # Giảm từ 256 -> 128

iron_yoke.nu_network_config.hidden_layers = 3                 # Giảm từ 3 -> 2 layers
iron_yoke.nu_network_config.hidden_neurons = 128               # Giảm từ 64 -> 32 neurons
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

magnet_left.training_config.number_of_interior_points = 2000
magnet_left.training_config.number_of_surface_points = 1000

magnet_left.network_config.hidden_layers = 4                 # Giảm từ 4 -> 3 layers
magnet_left.network_config.hidden_neurons = 64                # Giảm từ 64 -> 48 neurons
magnet_left.network_config.activation_function = nn.Mish()
magnet_left.network_config.fourier_features = False
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

magnet_right.training_config.number_of_interior_points = 2000
magnet_right.training_config.number_of_surface_points = 1000

magnet_right.network_config.hidden_layers = 4                  # Giảm từ 4 -> 3 layers
magnet_right.network_config.hidden_neurons = 64                # Giảm từ 64 -> 48 neurons
magnet_right.network_config.activation_function = nn.Mish()
magnet_right.network_config.fourier_features = False
magnet_right.network_config.fourier_sigma = 1.0
magnet_right.network_config.fourier_mapping_size = 64

magnet_right.build_domain()

# --- MIỀN 4: KHÔNG KHÍ NỀN (Background Air Domain) ---
bg_training_cfg = SimpleNamespace(
    pde_weight=1.0,
    interface_weight=1.0,
    interface_potential_weight=1.0,
    interface_bn_weight=1.0,
    interface_ht_weight=1.0,
    pde_fading_beta=80.0,
    sampling_density_multiplier=1.0,
    number_of_interior_points=15000,                            # Giảm nhẹ điểm nền từ 8,000 -> 5,000
    number_of_surface_points=0
)
bg_network_cfg = SimpleNamespace(
    hidden_layers=4,                                           # Giảm từ 4 -> 3 layers
    hidden_neurons=128,                                         # Giảm từ 64 -> 48 neurons
    activation_function=nn.Mish(),
    fourier_features=False,
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

print("[*] Bắt đầu tiến trình huấn luyện tối ưu tốc độ...")
model.execute_training_process()

print("[*] Xuất biểu đồ trường kết quả cuối cùng...")
model.plot_fields(
    resolution=200,
    save_path=f"{output_dir}/fpi_final_fields.png",
    show=False,
    figsize=(12, 10)
)
print(f"[✔] HOÀN THÀNH! Kết quả và animation đã lưu tại: {output_dir}/")