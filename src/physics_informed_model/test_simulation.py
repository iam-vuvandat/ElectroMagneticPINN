import os
import sys
from types import SimpleNamespace

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.insert(0, current_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import numpy as np
import matplotlib.pyplot as plt
from geometry_engine.segment.segment import Segment
from electro_magnetic_pinn import ElectroMagneticPINN

def main():
    print("version 5.3 - U-Magnet with 100% Full Overlap Corners")
    
    model = ElectroMagneticPINN()
    
    new_sampler_config = SimpleNamespace(
        x_boundaries_tuple=(-0.08, 0.08),  
        y_boundaries_tuple=(-0.08, 0.08)
    )
    
    new_pinn_config = SimpleNamespace(
        hidden_neurons=256,                
        activation_function=nn.SiLU()      
    )
    
    model.update_configuration(
        sampler_config=new_sampler_config,
        pinn_config=new_pinn_config
    )
    
    # CHÂN TRÁI
    left_leg = Segment(
        outline=[[-0.04, -0.03], [-0.02, -0.03], [-0.02, 0.03], [-0.04, 0.03]],
        material="u_left", 
        relative_permeability=1.05, 
        coercive=[0.0, 800000.0]  # Từ hóa hướng lên
    )

    # GÔNG TỪ KÉO DÀI TOÀN PHẦN (Phủ kín X từ -0.04 đến 0.04)
    # Tạo ra ô vuông giao nhau 20x20mm ở mỗi góc giúp Vector tự uốn cong 45 độ mượt mà
    yoke = Segment(
        outline=[[-0.04, -0.03], [0.04, -0.03], [0.04, -0.01], [-0.04, -0.01]],
        material="u_yoke", 
        relative_permeability=1.05, 
        coercive=[-800000.0, 0.0] # Từ hóa hướng sang trái
    )
    
    # CHÂN PHẢI
    right_leg = Segment(
        outline=[[0.02, -0.03], [0.04, -0.03], [0.04, 0.03], [0.02, 0.03]],
        material="u_right", 
        relative_permeability=1.05, 
        coercive=[0.0, -800000.0] # Từ hóa hướng xuống
    )

    # Nạp độc lập các segment
    model.geometry_engine_instance.add_segment(left_leg)
    model.geometry_engine_instance.add_segment(yoke)
    model.geometry_engine_instance.add_segment(right_leg)

    xb = model.sampler_config.x_boundaries_tuple
    yb = model.sampler_config.y_boundaries_tuple
    model.geometry_engine_instance.plot_problem_definition(
        x_boundaries_tuple=xb, y_boundaries_tuple=yb, resolution=120
    )

    model.execute_training_process(
        number_of_uniform_points=15000,    
        number_of_interface_points=5000,   
        distance_threshold=0.005,
        epochs_adam=4000,
        epochs_lbfgs=1000
    )
    
    resolution = 120
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(xy_points_tensor)
    
    A_z_grid = A_z_pred.numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.numpy().reshape(resolution, resolution)
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
    
    plt.show()

if __name__ == '__main__':
    main()
