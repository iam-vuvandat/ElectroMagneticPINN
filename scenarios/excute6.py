import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn

try:
    import ansys.aedt.core
except ImportError:
    if "google.colab" in sys.modules:
        print("[*] Installing pyaedt library on Colab...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyaedt"])

re_simulation_fem = False
re_train_pinn = True

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
pinn_module_path = os.path.join(parent_directory, "src", "physics_informed_model")

if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)
if pinn_module_path not in sys.path:
    sys.path.insert(0, pinn_module_path)

from src.ansys_aedt.maxwell_2d.toy_model_1.toy_model_1 import ToyModel1
from electro_magnetic_pinn import ElectroMagneticPINN
from src.ansys_aedt.maxwell_2d.toy_model_1.plot_flux_density import ANSYS_FEM_HEX_COLORS
from checkpoint_engine.save_checkpoint import resolve_checkpoint_filepath

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5
})


def main():
    output_dir = Path(parent_directory) / "figure_high_precision_simulation_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "toymodel1_relobralo"

    yoke_outline = [[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    magnet_left_outline = [[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]]
    magnet_right_outline = [[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]]
    region_outline = [[-0.12, -0.12], [0.12, -0.12], [0.12, 0.12], [-0.12, 0.12]]

    magnet_left_coercive_dir = (0, 1)
    magnet_right_coercive_dir = (0, -1)

    magnet_material_name = "NdFe30"
    yoke_material_name = "steel_1008"

    print("[1/4] ANSYS MAXWELL FEM SIMULATION PROCESSING")
    toy_model = ToyModel1()
    fem_config = toy_model.configuration

    fem_config.name = "toy_model_1"
    fem_config.version = "2025.2"
    fem_config.solution_type = "Magnetostatic"

    fem_config.geometry.yoke.outline = yoke_outline
    fem_config.geometry.magnet_left.outline = magnet_left_outline
    fem_config.geometry.magnet_left.coercive_direction = magnet_left_coercive_dir
    fem_config.geometry.magnet_right.outline = magnet_right_outline
    fem_config.geometry.magnet_right.coercive_direction = magnet_right_coercive_dir
    fem_config.geometry.region.outline = region_outline

    fem_config.materials.yoke_material = yoke_material_name
    fem_config.materials.magnet_material_left = "NdFe30 - Copy"
    fem_config.materials.magnet_material_right = "NdFe30 - Copy (2)"
    fem_config.materials.region_material = "vacuum"

    fem_config.analysis.setup_name = "Setup1"
    fem_config.analysis.max_passes = 10
    fem_config.analysis.min_passes = 2
    fem_config.analysis.percent_error = 1.0
    fem_config.analysis.max_element_length = "0.025mm"
    
    fld_path = Path(parent_directory) / "src" / "ansys_aedt" / "maxwell_2d" / "toy_model_1" / "flux_density.fld"
    fem_config.analysis.b_fld_path = str(fld_path)
    fem_config.analysis.grid_min = ["-120mm", "-120mm", "0mm"]
    fem_config.analysis.grid_max = ["120mm", "120mm", "0mm"]
    fem_config.analysis.grid_step = ["1mm", "1mm", "0mm"]

    if re_simulation_fem:
        print("[*] re_simulation_fem = True: Creating model and running FEM simulation on Ansys Maxwell...")
        toy_model.create_model()
        toy_model.run_simulation()
        toy_model.export_flux_density(export_path=str(fld_path))
    else:
        print("[*] re_simulation_fem = False: Reading FEM field data directly from .fld file...")
        if not fld_path.exists():
            raise FileNotFoundError(f"FEM field data file not found at: {fld_path}. Please set re_simulation_fem = True to simulate.")
        toy_model.export_flux_density(export_path=str(fld_path))

    field_record = toy_model.record.flux_density_field
    fem_x_mm = field_record.x
    fem_y_mm = field_record.y
    fem_b_mag = field_record.b_mag

    print("[2/4] PINN MODEL SETUP (ELECTROMAGNETIC PINN WITH GLOBAL RELOBRALO)")
    pinn_model = ElectroMagneticPINN()
    sim_domain = pinn_model.simulation_domain_engine_instance

    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 960000.0
    sim_domain.physics_config.scale_nu0 = 795774.715459
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    pinn_model.sampler_config.dynamic_resampling = True
    pinn_model.sampler_config.resample_interval = 10
    pinn_model.sampler_config.resample_points_before_lbfgs_stage = True

    pinn_model.visualization_config.active = True
    pinn_model.visualization_config.display_colab_frames = True
    pinn_model.visualization_config.update_interval = 200

    pinn_model.training_config.enable_relobralo = True
    pinn_model.training_config.relobralo_alpha = 0.999
    pinn_model.training_config.relobralo_temperature = 0.1
    pinn_model.training_config.relobralo_rho = 0.999

    pinn_model.training_config.enable_linear_warmup_stage = True
    pinn_model.training_config.enable_nonlinear_adaptation_stage = True
    pinn_model.training_config.enable_lbfgs_refinement_stage = True

    pinn_model.training_config.enable_lbfgs_after_warmup = True
    pinn_model.training_config.lbfgs_warmup_iterations = 1000

    pinn_model.training_config.enable_lbfgs_after_fpi_step = True
    pinn_model.training_config.lbfgs_fpi_step_iterations = 1000

    epochs_warmup_target = 5000
    fpi_iterations = 6
    epochs_per_fpi = 5000

    pinn_model.training_config.fpi_outer_iterations = fpi_iterations
    pinn_model.training_config.fpi_alpha_damping = 0.8
    pinn_model.training_config.fpi_tolerance = 1e-6

    pinn_model.training_config.number_of_adam_epochs = epochs_warmup_target + (fpi_iterations * epochs_per_fpi)
    pinn_model.training_config.linear_warmup_epochs_ratio = epochs_warmup_target / pinn_model.training_config.number_of_adam_epochs

    pinn_model.training_config.adam_learning_rate = 1e-3
    pinn_model.training_config.target_loss_threshold = 1e-7
    pinn_model.training_config.adam_eta_min = 1e-6
    pinn_model.training_config.grad_clip_max_norm = 1.0
    pinn_model.training_config.max_non_finite_streak = 5
    pinn_model.training_config.adam_log_interval = 10

    pinn_model.training_config.pretrain_nu_epochs = 3000
    pinn_model.training_config.pretrain_nu_lr = 5e-3
    pinn_model.training_config.fit_nu_epochs = 3000
    pinn_model.training_config.fit_nu_lr = 3e-3

    pinn_model.training_config.number_of_lbfgs_iterations = 4000
    pinn_model.training_config.lbfgs_learning_rate = 0.5





    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name=yoke_material_name,
        outline=yoke_outline
    )
    iron_yoke.physics_config.relative_permeability = 1000.0
    iron_yoke.physics_config.coercive = [0.0, 0.0]
    iron_yoke.physics_config.current = 0.0
    iron_yoke.physics_config.is_nonlinear = True

    iron_yoke.training_config.pde_fading_beta = 80.0
    iron_yoke.training_config.sampling_density_multiplier = 1.0
    iron_yoke.training_config.number_of_interior_points = 1800
    iron_yoke.training_config.number_of_surface_points = 700

    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 128
    iron_yoke.network_config.activation_function = nn.Mish()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.3
    iron_yoke.network_config.fourier_mapping_size = 128

    iron_yoke.nu_network_config.hidden_layers = 4
    iron_yoke.nu_network_config.hidden_neurons = 64
    iron_yoke.nu_network_config.activation_function = nn.Mish()
    iron_yoke.nu_network_config.fourier_features = True
    iron_yoke.nu_network_config.fourier_sigma = 1.0
    iron_yoke.nu_network_config.fourier_mapping_size = 32
    iron_yoke.build_domain()





    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left",
        material_name=magnet_material_name,
        outline=magnet_left_outline,
        coercive_direction=magnet_left_coercive_dir
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.number_of_interior_points = 400
    magnet_left.training_config.number_of_surface_points = 200
    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.Mish()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.0
    magnet_left.network_config.fourier_mapping_size = 32
    magnet_left.build_domain()

    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right",
        material_name=magnet_material_name,
        outline=magnet_right_outline,
        coercive_direction=magnet_right_coercive_dir
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.number_of_interior_points = 400
    magnet_right.training_config.number_of_surface_points = 200
    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.Mish()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.0
    magnet_right.network_config.fourier_mapping_size = 32
    magnet_right.build_domain()

    bg_training_cfg = SimpleNamespace(
        pde_fading_beta=80.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=2000,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=256,
        activation_function=nn.Mish(),
        fourier_features=True,
        fourier_sigma=1.0,
        fourier_mapping_size=120
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    pinn_model.update_electromagnetic_pinn()

    if re_train_pinn:
        print("[*] re_train_pinn = True: Training PINN model with global ReLoBRaLo...")
        pinn_model.execute_training_process()
        print(f"[*] Saving PINN checkpoint: '{checkpoint_name}'...")
        pinn_model.save_checkpoint(checkpoint_name)
        
        if "google.colab" in sys.modules:
            checkpoint_path = resolve_checkpoint_filepath(checkpoint_name)
            if checkpoint_path.exists():
                print(f"[*] Google Colab environment detected. Preparing download: {checkpoint_path}...")
                from google.colab import files
                files.download(str(checkpoint_path))
            else:
                print(f"[!] Checkpoint file not found for download: {checkpoint_path}")
    else:
        print(f"[*] re_train_pinn = False: Loading existing PINN checkpoint '{checkpoint_name}'...")
        pinn_model.load_checkpoint(checkpoint_name)

    print("\n==================================================")
    print("[3/4] COMPUTING PINN B-FIELD ON SYNCHRONIZED GRID")
    print("==================================================")
    res_x = len(fem_x_mm)
    res_y = len(fem_y_mm)
    x_m = fem_x_mm / 1000.0
    y_m = fem_y_mm / 1000.0
    X_m, Y_m = np.meshgrid(x_m, y_m)

    eval_pts = torch.tensor(
        np.column_stack((X_m.ravel(), Y_m.ravel())),
        dtype=torch.float32,
        device=pinn_model.computation_device
    ).requires_grad_(True)

    _, B_x_pinn, B_y_pinn = pinn_model.evaluate_fields(eval_pts, return_numpy=True)
    pinn_b_mag = np.sqrt(B_x_pinn**2 + B_y_pinn**2).reshape(res_y, res_x)

    delta_b_mag = np.abs(pinn_b_mag - fem_b_mag)
    fem_b_max = np.max(fem_b_mag)
    norm_error_percent = (delta_b_mag / fem_b_max) * 100.0

    print("\n==================================================")
    print("[4/4] PLOTTING AND COMPARING FEM VS PINN (3 PANELS)")
    print("==================================================")
    cmap_ansys = mcolors.LinearSegmentedColormap.from_list("ansys_fem_interp", ANSYS_FEM_HEX_COLORS, N=100)
    levels = np.linspace(0.0, 2.0, 101)
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap_ansys.N, clip=True)

    fig, axs = plt.subplots(1, 3, figsize=(21, 6), dpi=300)
    extent = [fem_x_mm.min(), fem_x_mm.max(), fem_y_mm.min(), fem_y_mm.max()]

    im0 = axs[0].imshow(
        fem_b_mag,
        extent=extent,
        origin="lower",
        cmap=cmap_ansys,
        norm=norm,
        aspect="equal"
    )
    axs[0].set_title(r"Ansys Maxwell 2D FEM $|B|$ (Tesla)", fontweight="bold")
    axs[0].set_xlabel("X (mm)")
    axs[0].set_ylabel("Y (mm)")
    cbar0 = fig.colorbar(im0, ax=axs[0], pad=0.02)
    cbar0.set_label(r"$|B|$ (T)")

    im1 = axs[1].imshow(
        pinn_b_mag,
        extent=extent,
        origin="lower",
        cmap=cmap_ansys,
        norm=norm,
        aspect="equal"
    )
    axs[1].set_title(r"ElectroMagneticPINN $|B|$ (Tesla)", fontweight="bold")
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Y (mm)")
    cbar1 = fig.colorbar(im1, ax=axs[1], pad=0.02)
    cbar1.set_label(r"$|B|$ (T)")

    im2 = axs[2].imshow(
        norm_error_percent,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=10.0,
        aspect="equal"
    )
    axs[2].set_title(r"Normalized Error (%) [$\Delta|B| / B_{\mathrm{max}}$]", fontweight="bold")
    axs[2].set_xlabel("X (mm)")
    axs[2].set_ylabel("Y (mm)")
    cbar2 = fig.colorbar(im2, ax=axs[2], pad=0.02)
    cbar2.set_label("Normalized Error (%)")

    plt.tight_layout()
    save_fig_path = output_dir / "fem_vs_pinn_bmap_comparison_relobralo.png"
    plt.savefig(save_fig_path, dpi=300)
    print(f"[*] Saved comparison plot to: {save_fig_path}")

    print("[*] Displaying figure...")
    plt.show()


if __name__ == "__main__":
    main()
