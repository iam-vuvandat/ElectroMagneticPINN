import os
import sys
import shutil
from types import SimpleNamespace

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
pinn_module_path = os.path.join(parent_directory, 'src', 'physics_informed_model')
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)
if pinn_module_path not in sys.path:
    sys.path.insert(0, pinn_module_path)

import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from src.physics_informed_model.electro_magnetic_pinn import ElectroMagneticPINN

output_dir = "figure_high_precision_simulation_test"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

model = ElectroMagneticPINN()
sim_domain = model.simulation_domain_engine_instance

re_train_model = False
checkpoint_name = "toymodel1"

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

epochs_warmup_target = 1000
fpi_iterations = 4
epochs_per_fpi = 1000

model.training_config.fpi_outer_iterations = fpi_iterations
model.training_config.fpi_alpha_damping = 0.5
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

vis_cfg = model.visualization_config
vis_cfg.active = True
vis_cfg.display_colab_frames = True
vis_cfg.update_interval = 10
vis_cfg.output_directory = output_dir
vis_cfg.resolution = 160

vis_cfg.generate_gif_animation = True
vis_cfg.gif_filename = "ansys_fem_training_process.gif"
vis_cfg.generate_mp4_video = True
vis_cfg.mp4_filename = "ansys_fem_training_process.mp4"
vis_cfg.animation_fps = 10.0
vis_cfg.gif_fps = 10.0
vis_cfg.mp4_codec = "libx264"
vis_cfg.mp4_quality = 8
vis_cfg.contour_levels = 80

vis_cfg.b_mag_scale = "linear"
vis_cfg.b_mag_cmap = "ansys_fem"
vis_cfg.b_mag_unit = "mTesla"
vis_cfg.b_mag_vmin = 0.0
vis_cfg.b_mag_vmax = 1800.0
vis_cfg.b_mag_number_of_division = 100

vis_cfg.res_scale = "log"
vis_cfg.res_vmin = 1e-4
vis_cfg.res_vmax = 10.0
vis_cfg.res_cmap = "inferno"
vis_cfg.res_discrete_levels = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

vis_cfg.mur_scale = "linear"
vis_cfg.mur_vmin = 0.0
vis_cfg.mur_vmax = 3000.0
vis_cfg.mur_cmap = "turbo"

vis_cfg.az_scale = "linear"
vis_cfg.az_vmin = None
vis_cfg.az_vmax = None
vis_cfg.az_cmap = "jet"

vis_cfg.by_scale = "linear"
vis_cfg.by_vmin = None
vis_cfg.by_vmax = None
vis_cfg.by_cmap = "coolwarm"

iron_yoke = sim_domain.create_material_domain(
    domain_name="iron_yoke",
    material_name="steel_1008",
    outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
)
iron_yoke.physics_config.relative_permeability = 1000.0
iron_yoke.physics_config.coercive = [0.0, 0.0]
iron_yoke.physics_config.current = 0.0
iron_yoke.physics_config.is_nonlinear = True

iron_yoke.training_config.pde_weight = 8.0
iron_yoke.training_config.pde_nonlinear_weight = 2.0
iron_yoke.training_config.interface_weight = 1.0
iron_yoke.training_config.interface_potential_weight = 800.0
iron_yoke.training_config.interface_bn_weight = 1000.0
iron_yoke.training_config.interface_ht_weight = 1200.0
iron_yoke.training_config.pde_fading_beta = 80.0
iron_yoke.training_config.sampling_density_multiplier = 1.0

iron_yoke.training_config.number_of_interior_points = 6000
iron_yoke.training_config.number_of_surface_points = 1500

iron_yoke.network_config.hidden_layers = 3
iron_yoke.network_config.hidden_neurons = 64
iron_yoke.network_config.activation_function = nn.Mish()
iron_yoke.network_config.fourier_features = True
iron_yoke.network_config.fourier_sigma = 1.3
iron_yoke.network_config.fourier_mapping_size = 64

iron_yoke.nu_network_config.hidden_layers = 2
iron_yoke.nu_network_config.hidden_neurons = 32
iron_yoke.nu_network_config.activation_function = nn.Mish()
iron_yoke.nu_network_config.fourier_features = True
iron_yoke.nu_network_config.fourier_sigma = 1.0
iron_yoke.nu_network_config.fourier_mapping_size = 16

iron_yoke.build_domain()

magnet_left = sim_domain.create_material_domain(
    domain_name="magnet_left",
    material_name="N30UH",
    outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
    coercive_direction=(0, 1)
)
magnet_left.physics_config.relative_permeability = 1.05
magnet_left.physics_config.current = 0.0
magnet_left.physics_config.is_nonlinear = False

magnet_left.training_config.number_of_interior_points = 800
magnet_left.training_config.number_of_surface_points = 400

magnet_left.network_config.hidden_layers = 3
magnet_left.network_config.hidden_neurons = 32
magnet_left.network_config.activation_function = nn.Mish()
magnet_left.network_config.fourier_features = True
magnet_left.network_config.fourier_sigma = 1.0
magnet_left.network_config.fourier_mapping_size = 32

magnet_left.build_domain()

magnet_right = sim_domain.create_material_domain(
    domain_name="magnet_right",
    material_name="N30UH",
    outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
    coercive_direction=(0, -1)
)
magnet_right.physics_config.relative_permeability = 1.05
magnet_right.physics_config.current = 0.0
magnet_right.physics_config.is_nonlinear = False

magnet_right.training_config.number_of_interior_points = 800
magnet_right.training_config.number_of_surface_points = 400

magnet_right.network_config.hidden_layers = 3
magnet_right.network_config.hidden_neurons = 32
magnet_right.network_config.activation_function = nn.Mish()
magnet_right.network_config.fourier_features = True
magnet_right.network_config.fourier_sigma = 1.0
magnet_right.network_config.fourier_mapping_size = 32

magnet_right.build_domain()

bg_training_cfg = SimpleNamespace(
    pde_weight=1.0,
    interface_weight=1.0,
    interface_potential_weight=800.0,
    interface_bn_weight=1000.0,
    interface_ht_weight=1200.0,
    pde_fading_beta=80.0,
    sampling_density_multiplier=1.0,
    number_of_interior_points=2500,
    number_of_surface_points=0
)
bg_network_cfg = SimpleNamespace(
    hidden_layers=3,
    hidden_neurons=32,
    activation_function=nn.Mish(),
    fourier_features=True,
    fourier_sigma=1.0,
    fourier_mapping_size=32
)
sim_domain.build_background_domain(
    domain_name="background",
    training_config=bg_training_cfg,
    network_config=bg_network_cfg
)

model.update_electromagnetic_pinn()

if re_train_model:
    print("[*] Starting PINN model training with Ansys FEM Colormap & Discrete Log Residual configuration...")
    model.execute_training_process()
    print(f"[*] Saving model checkpoint: '{checkpoint_name}'...")
    model.save_checkpoint(checkpoint_name)
else:
    print(f"[*] Loading model checkpoint: '{checkpoint_name}'...")
    model.load_checkpoint(checkpoint_name)

print("[*] Exporting final field distribution plots...")
model.plot_fields(
    resolution=200,
    save_path=f"{output_dir}/fpi_final_fields.png",
    show=False,
    figsize=(12, 10)
)
print(f"[*] COMPLETED! Results saved at: {output_dir}/")