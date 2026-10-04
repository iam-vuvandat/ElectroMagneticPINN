import os

def execute_refactor_to_composition():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # ---------------------------------------------------------
    # 1. CẬP NHẬT FILE ELECTRO_MAGNETIC_PINN.PY
    # ---------------------------------------------------------
    emp_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    emp_code = """import torch
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from geometry_engine.geometry import Geometry
from physics_domain.collocation_sampler import CollocationSampler

class ElectroMagneticPINN:
    def __init__(
        self, 
        geometry_config=None,
        sampler_config=None,
        pinn_config=None,
        training_config=None
    ):
        # 1. LƯU TRỮ CÁC CẤU HÌNH NHƯ LÀ THUỘC TÍNH CỦA CLASS MẸ
        self.geometry_config = geometry_config if geometry_config is not None else {}
        self.sampler_config = sampler_config if sampler_config is not None else {
            "x_boundaries": (-0.05, 0.05),
            "y_boundaries": (-0.05, 0.05)
        }
        self.pinn_config = pinn_config if pinn_config is not None else {}
        self.training_config = training_config if training_config is not None else {}
        
        # 2. KHỞI TẠO TRỰC TIẾP CÁC LỚP CON (LOẠI BỎ TIÊM PHỤ THUỘC)
        self.geometry_engine_instance = Geometry(**self.geometry_config)
        
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.get("x_boundaries", (-0.05, 0.05)),
            y_boundaries_tuple=self.sampler_config.get("y_boundaries", (-0.05, 0.05))
        )
        
        # Thiết lập hằng số quy chuẩn
        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = 795774.715459  # Hằng số chân không (vì đã bị gỡ khỏi Geometry)
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        # 3. TRUYỀN CẤU HÌNH VÀO CÁC LỚP LÕI THÔNG QUA **KWARGS
        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **self.pinn_config
        )
        
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            **self.training_config
        )

    def execute_training_process(self, number_of_uniform_points, number_of_interface_points, distance_threshold, epochs_adam, epochs_lbfgs):
        points_tensor = self.collocation_sampler_instance.generate_combined_points_tensor(
            geometry_object=self.geometry_engine_instance,
            number_of_uniform_points=number_of_uniform_points,
            number_of_interface_points=number_of_interface_points,
            distance_threshold=distance_threshold
        )
        
        physical_properties_dictionary = self.geometry_engine_instance.evaluate_global_physical_properties(points_tensor)
        
        print(f"--- Standard Adam Training ({epochs_adam} Epochs) ---")
        self.training_manager_instance.train_adam(
            epochs=epochs_adam,
            points_tensor=points_tensor,
            reluctivity_tensor=physical_properties_dictionary["reluctivity"],
            current_density_z_tensor=physical_properties_dictionary["current_density_z"],
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"],
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )
        
        print(f"--- L-BFGS Refinement ({epochs_lbfgs} Epochs) ---")
        self.training_manager_instance.train_lbfgs(
            epochs=epochs_lbfgs, 
            points_tensor=points_tensor, 
            reluctivity_tensor=physical_properties_dictionary["reluctivity"], 
            current_density_z_tensor=physical_properties_dictionary["current_density_z"], 
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"], 
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )

    def predict_magnetic_vector_potential(self, points_tensor):
        self.pinn_architecture_instance.eval()
        with torch.no_grad():
            A_z_star = self.pinn_architecture_instance(points_tensor)
        return A_z_star * self.A0

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor.requires_grad_(True)
        
        A_z_star = self.pinn_architecture_instance(points_tensor)
        A_z_phys = A_z_star * self.A0
        
        grad_A = torch.autograd.grad(
            outputs=A_z_phys,
            inputs=points_tensor,
            grad_outputs=torch.ones_like(A_z_phys),
            create_graph=False,
            retain_graph=False
        )[0]
        
        B_x = grad_A[:, 1:2]
        B_y = -grad_A[:, 0:1]
        
        return A_z_phys.detach(), B_x.detach(), B_y.detach()
"""

    # ---------------------------------------------------------
    # 2. CẬP NHẬT FILE TEST_SIMULATION.PY
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
from geometry_engine.segment.segment import Segment
from electro_magnetic_pinn import ElectroMagneticPINN

def main():
    # 1. ĐỊNH NGHĨA CÁC DICTIONARY CẤU HÌNH RÕ RÀNG
    sampler_config = {
        "x_boundaries": (-0.05, 0.05),
        "y_boundaries": (-0.05, 0.05)
    }
    
    pinn_config = {
        "hidden_layers": 4,
        "hidden_neurons": 64
    }
    
    training_config = {
        "lr_adam": 1e-3,
        "target_loss": 1e-3,
        "lbfgs_lr": 0.8,
        "lbfgs_tolerance_grad": 1e-8,
        "lbfgs_tolerance_change": 1e-10
    }

    # 2. KHỞI TẠO LỚP MẸ CHỈ VỚI CÁC CẤU HÌNH (Lớp mẹ sẽ tự xây dựng hệ thống con)
    model = ElectroMagneticPINN(
        sampler_config=sampler_config,
        pinn_config=pinn_config,
        training_config=training_config
    )
    
    # 3. THAO TÁC TRỰC TIẾP LÊN THUỘC TÍNH CON CỦA LỚP MẸ
    top_magnet_vertices = [
        [-0.03, 0.015], [0.03, 0.015], [0.03, 0.025], [-0.03, 0.025]
    ]
    top_magnet = Segment(outline=top_magnet_vertices).set_material_properties(
        material="top_magnet",
        relative_permeability=1.05,
        coercive=[800000.0, 0.0]
    )
    model.geometry_engine_instance.add_segment(top_magnet)

    bottom_magnet_vertices = [
        [-0.03, -0.025], [0.03, -0.025], [0.03, -0.015], [-0.03, -0.015]
    ]
    bottom_magnet = Segment(outline=bottom_magnet_vertices).set_material_properties(
        material="bottom_magnet",
        relative_permeability=1.05,
        coercive=[-800000.0, 0.0]
    )
    model.geometry_engine_instance.add_segment(bottom_magnet)

    # Lấy thông số từ sampler config để vẽ hình
    xb = model.sampler_config["x_boundaries"]
    yb = model.sampler_config["y_boundaries"]
    model.geometry_engine_instance.plot_problem_definition(
        x_boundaries_tuple=xb,
        y_boundaries_tuple=yb,
        resolution=100
    )

    # 4. KÍCH HOẠT QUÁ TRÌNH HUẤN LUYỆN
    model.execute_training_process(
        number_of_uniform_points=5000,
        number_of_interface_points=1500,
        distance_threshold=0.005,
        epochs_adam=4000,
        epochs_lbfgs=1000
    )
    
    # --- ĐOẠN MÃ VẼ BIỂU ĐỒ (Giữ nguyên) ---
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

    files_to_update = [
        (emp_file, emp_code),
        (test_file, test_code)
    ]
    
    print("Đang cấu trúc lại quan hệ Hợp thành (Composition) và Config...")
    for file_path, content in files_to_update:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + "\n")
        print(f"[+] Đã cập nhật: {os.path.basename(file_path)}")
        
    print("\nHOÀN TẤT! File main hiện tại cực kỳ gọn gàng với Configuration Dictionaries.")

if __name__ == "__main__":
    execute_refactor_to_composition()