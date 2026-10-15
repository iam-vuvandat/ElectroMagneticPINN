import os
import sys
import shutil

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
from simulation_domain_engine.material_domain.material_domain import MaterialDomain
from electro_magnetic_pinn import ElectroMagneticPINN

def main():
    output_dir = "figure_iron_yoke"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print("version 1.0 - Dual Magnets on Constant Permeability Iron Yoke Simulation")
    
    model = ElectroMagneticPINN()
    
    max_coercive_field = 800000.0
    model.simulation_domain_engine_instance.physics_config.scale_H0 = max_coercive_field * 1.2
    model.simulation_domain_engine_instance.physics_config.scale_L0 = 0.1 * 1.2
    
    model.simulation_domain_engine_instance.x_boundaries_tuple = (-0.1, 0.1)
    model.simulation_domain_engine_instance.y_boundaries_tuple = (-0.1, 0.1)
    
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_before_lbfgs = True
    
    model.training_config.epochs_adam = 4000
    model.training_config.epochs_lbfgs = 2000
    model.training_config.learning_rate_adam = 1e-3
    model.training_config.target_loss = 1e-7
    
    model.training_config.lbfgs_learning_rate = 0.8
    model.training_config.lbfgs_maximum_iterations = 2000
    model.training_config.lbfgs_maximum_evaluations = 2500
    model.training_config.lbfgs_tolerance_gradient = 1e-8
    model.training_config.lbfgs_tolerance_change = 1e-10
    model.training_config.lbfgs_history_size = 50
    
    model.visualization_config.active = True
    model.visualization_config.update_interval = 50
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "iron_yoke_training_process.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "iron_yoke_training_process.mp4"
    model.visualization_config.animation_fps = 5
    model.visualization_config.gif_fps = 5

    iron_yoke = MaterialDomain(domain_name="iron_yoke")
    iron_yoke.geometry_config.outline = [
        [-0.06, -0.03], 
        [0.06, -0.03], 
        [0.06, 0.0], 
        [-0.06, 0.0]
    ]
    iron_yoke.physics_config.relative_permeability = 1000.0
    iron_yoke.physics_config.coercive = [0.0, 0.0]
    iron_yoke.physics_config.current = 0.0
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_weight = 1.0
    iron_yoke.training_config.interface_potential_weight = 1000.0
    iron_yoke.training_config.interface_bn_weight = 350.0
    iron_yoke.training_config.interface_ht_weight = 350.0
    iron_yoke.training_config.pde_fading_beta = 90.0
    iron_yoke.training_config.number_of_interior_points = 2500
    iron_yoke.training_config.number_of_surface_points = 1200
    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128
    iron_yoke.build_domain()

    magnet_left = MaterialDomain(domain_name="magnet_left")
    magnet_left.geometry_config.outline = [
        [-0.05, 0.0], 
        [-0.005, 0.0], 
        [-0.005, 0.018], 
        [-0.05, 0.018]
    ]
    magnet_left.physics_config.relative_permeability = 1.05
    magnet_left.physics_config.coercive = [0.0, 800000.0]
    magnet_left.physics_config.current = 0.0
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_weight = 1.0
    magnet_left.training_config.interface_potential_weight = 1000.0
    magnet_left.training_config.interface_bn_weight = 350.0
    magnet_left.training_config.interface_ht_weight = 350.0
    magnet_left.training_config.pde_fading_beta = 90.0
    magnet_left.training_config.number_of_interior_points = 1500
    magnet_left.training_config.number_of_surface_points = 800
    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128
    magnet_left.build_domain()

    magnet_right = MaterialDomain(domain_name="magnet_right")
    magnet_right.geometry_config.outline = [
        [0.005, 0.0], 
        [0.05, 0.0], 
        [0.05, 0.018], 
        [0.005, 0.018]
    ]
    magnet_right.physics_config.relative_permeability = 1.05
    magnet_right.physics_config.coercive = [0.0, -800000.0]
    magnet_right.physics_config.current = 0.0
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_weight = 1.0
    magnet_right.training_config.interface_potential_weight = 1000.0
    magnet_right.training_config.interface_bn_weight = 350.0
    magnet_right.training_config.interface_ht_weight = 350.0
    magnet_right.training_config.pde_fading_beta = 90.0
    magnet_right.training_config.number_of_interior_points = 1500
    magnet_right.training_config.number_of_surface_points = 800
    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128
    magnet_right.build_domain()

    model.simulation_domain_engine_instance.add_material_domain(iron_yoke)
    model.simulation_domain_engine_instance.add_material_domain(magnet_left)
    model.simulation_domain_engine_instance.add_material_domain(magnet_right)

    bg_training_config = type('', (), {})()
    bg_training_config.pde_weight = 1.0
    bg_training_config.interface_weight = 1.0
    bg_training_config.pde_fading_beta = 90.0
    bg_training_config.sampling_density_multiplier = 1.0
    bg_training_config.number_of_interior_points = 7000
    bg_training_config.number_of_surface_points = 0

    bg_network_config = type('', (), {})()
    bg_network_config.hidden_layers = 4
    bg_network_config.hidden_neurons = 64
    bg_network_config.activation_function = nn.SiLU()
    bg_network_config.fourier_features = True
    bg_network_config.fourier_sigma = 1.2
    bg_network_config.fourier_mapping_size = 128

    model.simulation_domain_engine_instance.build_background_domain(
        domain_name="background",
        training_config=bg_training_config,
        network_config=bg_network_config
    )

    model.update_electromagnetic_pinn()

    xb = model.simulation_domain_engine_instance.x_boundaries_tuple
    yb = model.simulation_domain_engine_instance.y_boundaries_tuple
    
    model.simulation_domain_engine_instance.plot_problem_definition(resolution=100)

    model.execute_training_process()
    
    resolution = 180
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(xy_points_tensor)
    assert torch.all(torch.isfinite(A_z_pred)) and torch.all(torch.isfinite(B_x_pred)) and torch.all(torch.isfinite(B_y_pred)), "Trường từ tính toán không hữu hạn."
    
    A_z_grid = A_z_pred.cpu().numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.cpu().numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.cpu().numpy().reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)
    
    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    contour_az = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=60, cmap="jet")
    fig1.colorbar(contour_az, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
    axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
    axs[0, 0].set_aspect('equal')
    
    b_max_val = max(float(np.max(B_mag_grid)), 1e-3)
    levels_b = np.linspace(0.0, b_max_val, 60)
    contour_b = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=levels_b, cmap="jet", vmin=0.0, vmax=b_max_val)
    fig1.colorbar(contour_b, ax=axs[0, 1], label=r"$|B|$ (T)")
    axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_aspect('equal')
    
    contour_bx = axs[1, 0].contourf(X_grid, Y_grid, B_x_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_bx, ax=axs[1, 0], label=r"$B_x$ (T)")
    axs[1, 0].set_title(r"Magnetic Field Component ($B_x$)")
    axs[1, 0].set_aspect('equal')
    
    contour_by = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_by, ax=axs[1, 1], label=r"$B_y$ (T)")
    axs[1, 1].set_title(r"Magnetic Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')
    fig1.tight_layout()
    plt.savefig(f'{output_dir}/final_fields_grid.png', dpi=150)
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    contour_b_bg = ax2.contourf(X_grid, Y_grid, B_mag_grid, levels=levels_b, cmap="jet", alpha=0.35, vmin=0.0, vmax=b_max_val)
    fig2.colorbar(contour_b_bg, ax=ax2, label=r"$|B|$ (T)")
    
    step = 6
    B_sub = np.maximum(B_mag_grid[::step, ::step], 1e-12)
    ax2.quiver(
        X_grid[::step, ::step], Y_grid[::step, ::step],
        B_x_grid[::step, ::step] / B_sub, B_y_grid[::step, ::step] / B_sub,
        scale=35.0, color='black', pivot='mid'
    )
    ax2.set_title(r"Magnetic Flux Density Vectors ($\mathbf{B}$)")
    ax2.set_aspect('equal')
    fig2.tight_layout()
    
    plt.savefig(f'{output_dir}/final_results.png', dpi=150)
    plt.close('all')
    print(f'[*] Đã lưu kết quả mô phỏng vào {output_dir}/final_results.png và {output_dir}/final_fields_grid.png')

if __name__ == "__main__":
    main()