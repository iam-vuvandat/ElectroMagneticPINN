import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.insert(0, current_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_engine.segment.segment import Segment
from electro_magnetic_pinn import ElectroMagneticPINN
import shutil

def main():
    if os.path.exists('figure'):
        shutil.rmtree('figure')
    os.makedirs('figure')
    print("version 7.0 - Dual Straight I-Shaped Magnets Simulation")
    
    model = ElectroMagneticPINN()
    
    max_coercive_field = 800000.0
    model.physics_config.scale_H0 = max_coercive_field * 1.5
    model.physics_config.scale_L0 = 0.12
    
    model.sampler_config.x_boundaries_tuple = (-0.1, 0.1)
    model.sampler_config.y_boundaries_tuple = (-0.1, 0.1)
    
    model.sampler_config.number_of_uniform_points = 2000
    model.sampler_config.number_of_interface_points = 6000
    model.sampler_config.distance_threshold = 0.005
    
    model.pinn_config.hidden_layers = 6
    model.pinn_config.hidden_neurons = 64
    model.pinn_config.activation_function = nn.Tanh()
    
    model.training_config.epochs_adam = 4000
    # #model.training_config.epochs_lbfgs = 300 # Đã tự động loại bỏ
    model.training_config.learning_rate_adam = 1e-3
    model.training_config.target_loss = 0.0
    # model.training_config.lbfgs_learning_rate = 0.8 # Đã tự động loại bỏ
    # model.training_config.lbfgs_maximum_iterations = 1000 # Đã tự động loại bỏ
    # model.training_config.lbfgs_maximum_evaluations = 1250 # Đã tự động loại bỏ
    # model.training_config.lbfgs_tolerance_gradient = 1e-8 # Đã tự động loại bỏ
    # model.training_config.lbfgs_tolerance_change = 1e-10 # Đã tự động loại bỏ
    # model.training_config.lbfgs_history_size = 50 # Đã tự động loại bỏ
    
    model.visualization_config.active = True
    model.visualization_config.update_interval = 50 
    model.visualization_config.output_directory = "figure"
    model.visualization_config.resolution = 80
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.gif_fps = 15
    
    model.update_electromagnetic_pinn()
    
    i_magnet_left = Segment(
        outline=[[-0.04, -0.04], [-0.02, -0.04], [-0.02, 0.04], [-0.04, 0.04]],
        material="i_bar_left", 
        relative_permeability=1.05, 
        coercive=[0.0, 800000.0],
        steepness=0.1  # BỔ SUNG: Làm mượt ranh giới
    )

    i_magnet_right = Segment(
        outline=[[0.02, -0.04], [0.04, -0.04], [0.04, 0.04], [0.02, 0.04]],
        material="i_bar_right", 
        relative_permeability=1.05, 
        coercive=[0.0, -800000.0],
        steepness=0.1  # BỔ SUNG: Làm mượt ranh giới
    )

    model.geometry_engine_instance.add_segment(i_magnet_left)
    model.geometry_engine_instance.add_segment(i_magnet_right)

    xb = model.sampler_config.x_boundaries_tuple
    yb = model.sampler_config.y_boundaries_tuple
    
    model.geometry_engine_instance.plot_problem_definition(
        x_boundaries_tuple=xb, y_boundaries_tuple=yb, resolution=120
    )

    model.execute_training_process()
    
    resolution = 120
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(xy_points_tensor)
    
    A_z_grid = A_z_pred.cpu().numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.cpu().numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.cpu().numpy().reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid**2 + B_y_grid**2)
    
    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    contour_az = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=60, cmap="jet")
    fig1.colorbar(contour_az, ax=axs[0, 0], label="A_z (Wb/m)")
    axs[0, 0].set_title("Magnetic Vector Potential ($A_z$)")
    axs[0, 0].set_aspect('equal')
    
    contour_b = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=60, cmap="rainbow")
    fig1.colorbar(contour_b, ax=axs[0, 1], label="|B| (T)")
    axs[0, 1].set_title("Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_aspect('equal')
    
    contour_bx = axs[1, 0].contourf(X_grid, Y_grid, B_x_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_bx, ax=axs[1, 0], label="B_x (T)")
    axs[1, 0].set_title("Magnetic Field Component ($B_x$)")
    axs[1, 0].set_aspect('equal')
    
    contour_by = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_by, ax=axs[1, 1], label="B_y (T)")
    axs[1, 1].set_title("Magnetic Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    contour_b_bg = ax2.contourf(X_grid, Y_grid, B_mag_grid, levels=60, cmap="rainbow", alpha=0.4)
    fig2.colorbar(contour_b_bg, ax=ax2, label="|B| (T)")
    
    step = 4
    ax2.quiver(X_grid[::step, ::step], Y_grid[::step, ::step], B_x_grid[::step, ::step], B_y_grid[::step, ::step], color='black', pivot='mid')
    ax2.set_title("Magnetic Flux Density Vectors (B)")
    ax2.set_aspect('equal')
    fig2.tight_layout()
    
    plt.savefig('figure/final_results.png', dpi=150)
    plt.close('all')
    print('[*] Đã lưu kết quả mô phỏng vào final_results.png')

if __name__ == '__main__':
    main()