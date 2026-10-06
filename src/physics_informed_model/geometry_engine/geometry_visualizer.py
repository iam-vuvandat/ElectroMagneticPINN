import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

def plot_geometry_problem(geometry_instance, x_boundaries_tuple, y_boundaries_tuple, resolution=100):
    x_coords = np.linspace(x_boundaries_tuple[0], x_boundaries_tuple[1], resolution)
    y_coords = np.linspace(y_boundaries_tuple[0], y_boundaries_tuple[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    
    sdf_values_tensor = geometry_instance.compute_global_signed_distance_field(xy_points_tensor)
    physical_properties_dictionary = geometry_instance.evaluate_global_physical_properties(xy_points_tensor)
    
    sdf_grid = sdf_values_tensor.numpy().reshape(resolution, resolution)
    
    reluctivity_tensor = physical_properties_dictionary["reluctivity"]
    
    # SỬ DỤNG HẰNG SỐ TOÀN CỤC ĐÃ ĐƯỢC IMPORT
    mu_r_tensor = VACUUM_RELUCTIVITY / reluctivity_tensor
    mu_r_grid = mu_r_tensor.numpy().reshape(resolution, resolution)
    
    hx_tensor = physical_properties_dictionary["coercive_field_x"]
    hy_tensor = physical_properties_dictionary["coercive_field_y"]
    hc_magnitude_tensor = torch.sqrt(hx_tensor**2 + hy_tensor**2)
    hc_grid = hc_magnitude_tensor.numpy().reshape(resolution, resolution)
    
    jz_tensor = physical_properties_dictionary["current_density_z"]
    jz_grid = jz_tensor.numpy().reshape(resolution, resolution)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    contour_sdf = axs[0, 0].contourf(X_grid, Y_grid, sdf_grid, levels=50, cmap="coolwarm")
    axs[0, 0].contour(X_grid, Y_grid, sdf_grid, levels=[0.0], colors="black", linewidths=1.5)
    fig.colorbar(contour_sdf, ax=axs[0, 0])
    axs[0, 0].set_title("Signed Distance Field (SDF)")
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].set_aspect('equal')
    
    contour_mur = axs[0, 1].contourf(X_grid, Y_grid, mu_r_grid, levels=50, cmap="viridis")
    fig.colorbar(contour_mur, ax=axs[0, 1])
    axs[0, 1].set_title("Relative Permeability (mu_r)")
    axs[0, 1].set_xlabel("x (m)")
    axs[0, 1].set_ylabel("y (m)")
    axs[0, 1].set_aspect('equal')
    
    contour_hc = axs[1, 0].contourf(X_grid, Y_grid, hc_grid, levels=50, cmap="plasma")
    fig.colorbar(contour_hc, ax=axs[1, 0])
    axs[1, 0].set_title("Magnetization Magnitude (Hc)")
    axs[1, 0].set_xlabel("x (m)")
    axs[1, 0].set_ylabel("y (m)")
    axs[1, 0].set_aspect('equal')
    
    contour_jz = axs[1, 1].contourf(X_grid, Y_grid, jz_grid, levels=50, cmap="inferno")
    fig.colorbar(contour_jz, ax=axs[1, 1])
    axs[1, 1].set_title("Current Density (Jz)")
    axs[1, 1].set_xlabel("x (m)")
    axs[1, 1].set_ylabel("y (m)")
    axs[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('geometry_plot.png', dpi=150)
    plt.close(fig)
    print('[*] Đã lưu biểu đồ hình học vào geometry_plot.png')
