import torch
import numpy as np
import matplotlib.pyplot as plt
from geometry_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

def display_geometry_problem(geometry_instance, x_boundaries_tuple, y_boundaries_tuple, resolution=100):
    x_coords = np.linspace(x_boundaries_tuple[0], x_boundaries_tuple[1], resolution)
    y_coords = np.linspace(y_boundaries_tuple[0], y_boundaries_tuple[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    
    # Đánh giá thông số vật lý tại các điểm trên lưới
    physical_properties = geometry_instance.evaluate_global_physical_properties(xy_points_tensor)
    
    # Trích xuất phân loại vật liệu (Segment ID)
    mat_id_tensor = physical_properties["material_classification"]
    mat_id_grid = mat_id_tensor.numpy().reshape(resolution, resolution)
    
    # Trích xuất độ từ thẩm tương đối
    reluctivity_tensor = physical_properties["reluctivity"]
    mu_r_tensor = VACUUM_RELUCTIVITY / reluctivity_tensor
    mu_r_grid = mu_r_tensor.numpy().reshape(resolution, resolution)
    
    # Trích xuất độ lớn lực kháng từ (Magnetization)
    hx_tensor = physical_properties["coercive_field_x"]
    hy_tensor = physical_properties["coercive_field_y"]
    hc_magnitude_tensor = torch.sqrt(hx_tensor**2 + hy_tensor**2)
    hc_grid = hc_magnitude_tensor.numpy().reshape(resolution, resolution)
    
    # Trích xuất mật độ dòng điện (Jz)
    jz_tensor = physical_properties["current_density_z"]
    jz_grid = jz_tensor.numpy().reshape(resolution, resolution)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Đồ thị 1: Sơ đồ vật liệu (Segment ID)
    contour_mat = axs[0, 0].contourf(X_grid, Y_grid, mat_id_grid, cmap="tab10", levels=geometry_instance.get_segment_count() + 1)
    fig.colorbar(contour_mat, ax=axs[0, 0], label="Segment ID")
    axs[0, 0].set_title("Material Layout (Segment IDs)")
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].set_aspect('equal')
    
    # Đồ thị 2: Độ từ thẩm
    contour_mur = axs[0, 1].contourf(X_grid, Y_grid, mu_r_grid, levels=50, cmap="viridis")
    fig.colorbar(contour_mur, ax=axs[0, 1], label="$\mu_r$")
    axs[0, 1].set_title("Relative Permeability ($\mu_r$)")
    axs[0, 1].set_xlabel("x (m)")
    axs[0, 1].set_ylabel("y (m)")
    axs[0, 1].set_aspect('equal')
    
    # Đồ thị 3: Độ từ hóa
    contour_hc = axs[1, 0].contourf(X_grid, Y_grid, hc_grid, levels=50, cmap="plasma")
    fig.colorbar(contour_hc, ax=axs[1, 0], label="$H_c$ (A/m)")
    axs[1, 0].set_title("Magnetization Magnitude ($H_c$)")
    axs[1, 0].set_xlabel("x (m)")
    axs[1, 0].set_ylabel("y (m)")
    axs[1, 0].set_aspect('equal')
    
    # Đồ thị 4: Mật độ dòng điện
    contour_jz = axs[1, 1].contourf(X_grid, Y_grid, jz_grid, levels=50, cmap="inferno")
    fig.colorbar(contour_jz, ax=axs[1, 1], label="$J_z$ (A/m$^2$)")
    axs[1, 1].set_title("Current Density ($J_z$)")
    axs[1, 1].set_xlabel("x (m)")
    axs[1, 1].set_ylabel("y (m)")
    axs[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show() 