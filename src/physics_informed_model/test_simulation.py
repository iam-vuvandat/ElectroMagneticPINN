import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.insert(0, current_directory)

import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import numpy as np
import matplotlib.pyplot as plt
from geometry_engine.geometry import Geometry
from geometry_engine.segment.segment import Segment
from physics_domain.collocation_sampler import CollocationSampler
from electro_magnetic_pinn import ElectroMagneticPINN

def main():
    geometry_instance = Geometry()
    
    top_magnet_vertices = [
        [-0.03, 0.015], [0.03, 0.015], [0.03, 0.025], [-0.03, 0.025]
    ]
    # Cập nhật theo giao diện Segment mới
    top_magnet = Segment(outline=top_magnet_vertices).set_material_properties(
        material="top_magnet",
        relative_permeability=1.05,
        coercive=[800000.0, 0.0]
    )
    geometry_instance.add_segment(top_magnet)

    bottom_magnet_vertices = [
        [-0.03, -0.025], [0.03, -0.025], [0.03, -0.015], [-0.03, -0.015]
    ]
    # Cập nhật theo giao diện Segment mới
    bottom_magnet = Segment(outline=bottom_magnet_vertices).set_material_properties(
        material="bottom_magnet",
        relative_permeability=1.05,
        coercive=[-800000.0, 0.0]
    )
    geometry_instance.add_segment(bottom_magnet)
    
    collocation_sampler_instance = CollocationSampler(
        x_boundaries_tuple=(-0.05, 0.05),
        y_boundaries_tuple=(-0.05, 0.05)
    )

    geometry_instance.plot_problem_definition(
        x_boundaries_tuple=(-0.05, 0.05),
        y_boundaries_tuple=(-0.05, 0.05),
        resolution=100
    )

    model = ElectroMagneticPINN(
        geometry_engine_instance=geometry_instance,
        collocation_sampler_instance=collocation_sampler_instance,
        lr_adam=1e-3,
        lbfgs_lr=0.8,
        lbfgs_max_iter=1000,
        lbfgs_max_eval=1250
    )
    
    model.execute_training_process(
        number_of_uniform_points=5000,
        number_of_interface_points=1500,
        distance_threshold=0.005,
        epochs_adam=4000,
        epochs_lbfgs=1000
    )
    
    resolution = 120
    x_coords = np.linspace(-0.05, 0.05, resolution)
    y_coords = np.linspace(-0.05, 0.05, resolution)
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

if __name__ == "__main__":
    main()
