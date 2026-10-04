import os

def execute_update():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # ---------------------------------------------------------
    # 1. CẬP NHẬT FILE TEST_SIMULATION.PY
    # ---------------------------------------------------------
    test_file = os.path.join(project_root, 'test_simulation.py')
    test_code = """import os
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
"""

    # ---------------------------------------------------------
    # 2. CẬP NHẬT FILE GEOMETRY.PY
    # ---------------------------------------------------------
    geom_file = os.path.join(project_root, 'geometry_engine', 'geometry.py')
    geom_code = """import torch
from geometry_engine.segment.segment import Segment
from geometry_engine.global_signed_distance_field import compute_global_signed_distance_field
from geometry_engine.global_physical_properties_evaluation import evaluate_global_physical_properties
from geometry_engine.geometry_visualizer import plot_geometry_problem

class Geometry:
    def __init__(self):
        self.segments_list = []
        self.vacuum_reluctivity = 795774.715459

    def add_segment(self, segment_object):
        # Tự động khởi tạo thông số nội tại của Segment mới trước khi lưu
        segment_object.compute_section_area()
        segment_object.compute_current_density()
        segment_object.calculate_penetrating_steepness()
        
        self.segments_list.append(segment_object)
        return self

    def compute_global_signed_distance_field(self, points_tensor):
        return compute_global_signed_distance_field(self.segments_list, points_tensor)

    def evaluate_global_physical_properties(self, points_tensor):
        return evaluate_global_physical_properties(self.segments_list, points_tensor, self.vacuum_reluctivity)

    def plot_problem_definition(self, x_boundaries_tuple, y_boundaries_tuple, resolution=100):
        plot_geometry_problem(self, x_boundaries_tuple, y_boundaries_tuple, resolution)
"""

    # ---------------------------------------------------------
    # 3. CẬP NHẬT FILE GLOBAL_PHYSICAL_PROPERTIES_EVALUATION.PY
    # ---------------------------------------------------------
    prop_file = os.path.join(project_root, 'geometry_engine', 'global_physical_properties_evaluation.py')
    prop_code = """import torch

def evaluate_global_physical_properties(segments_list, points_tensor, vacuum_reluctivity):
    number_of_points = points_tensor.shape[0]
    computation_device = points_tensor.device
    
    global_reluctivity_tensor = torch.full((number_of_points, 1), vacuum_reluctivity, dtype=torch.float32, device=computation_device)
    global_coercive_field_x_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_coercive_field_y_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_current_density_z_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_material_classification_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)

    material_index_counter = 1.0

    for segment_object in segments_list:
        signed_distance_field = segment_object.compute_signed_distance_field(points_tensor)
        
        # Áp dụng độ dốc động (Adaptive Steepness) của Segment hiện tại
        # Công thức giải ngược: s = (5000.0 - ideal_k) / 4960.0  => ideal_k = 5000.0 - s * 4960.0
        actual_k = 5000.0 - segment_object.steepness * 4960.0
        mask_smooth = torch.sigmoid(-actual_k * signed_distance_field).view(-1, 1)
        
        seg_reluctivity = segment_object.evaluate_reluctivity(points_tensor)
        hx_tensor, hy_tensor = segment_object.evaluate_magnetization_vector(points_tensor)
        seg_jz = segment_object.evaluate_current_density(points_tensor)
        
        global_reluctivity_tensor = global_reluctivity_tensor + mask_smooth * (seg_reluctivity - vacuum_reluctivity)
        global_coercive_field_x_tensor = global_coercive_field_x_tensor + mask_smooth * hx_tensor
        global_coercive_field_y_tensor = global_coercive_field_y_tensor + mask_smooth * hy_tensor
        global_current_density_z_tensor = global_current_density_z_tensor + mask_smooth * seg_jz
        
        global_material_classification_tensor = global_material_classification_tensor + mask_smooth * material_index_counter
        
        material_index_counter += 1.0

    return {
        "reluctivity": global_reluctivity_tensor,
        "coercive_field_x": global_coercive_field_x_tensor,
        "coercive_field_y": global_coercive_field_y_tensor,
        "current_density_z": global_current_density_z_tensor,
        "material_classification": global_material_classification_tensor
    }
"""

    # Tiến hành ghi file
    files_to_update = [
        (test_file, test_code),
        (geom_file, geom_code),
        (prop_file, prop_code)
    ]
    
    print("Đang cập nhật các tệp liên kết với lớp Segment mới...")
    for file_path, content in files_to_update:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + "\n")
        print(f"[+] Đã cập nhật: {os.path.basename(file_path)}")
        
    print("\nHOÀN TẤT! Hệ thống đã được đồng bộ với kiến trúc Segment mới.")

if __name__ == "__main__":
    execute_update()