import os

def execute_refactor_u_magnet_and_templates():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # ---------------------------------------------------------
    # 1. CẬP NHẬT FILE ELECTRO_MAGNETIC_PINN.PY (Cơ chế Template + Update)
    # ---------------------------------------------------------
    emp_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    emp_code = """import torch
import torch.nn as nn
from types import SimpleNamespace
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from geometry_engine.geometry import Geometry
from physics_domain.collocation_sampler import CollocationSampler

class ElectroMagneticPINN:
    def __init__(self):
        # 1. KHỞI TẠO CẤU HÌNH TEMPLATE MẶC ĐỊNH
        self.sampler_config = SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05)
        )
        
        self.pinn_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU()
        )
        
        self.training_config = SimpleNamespace(
            lr_adam=1e-3,
            target_loss=1e-3,
            lbfgs_lr=0.8,
            lbfgs_max_iter=1000,
            lbfgs_max_eval=1250,
            lbfgs_tolerance_grad=1e-8,
            lbfgs_tolerance_change=1e-10,
            lbfgs_history_size=50
        )
        
        # Geometry độc lập với các thông số vật lý nên chỉ cần tạo 1 lần
        self.geometry_engine_instance = Geometry()
        
        # Xây dựng hệ thống lần đầu
        self._build_system()

    def update_configuration(self, sampler_config=None, pinn_config=None, training_config=None):
        # 2. HÀM UPDATE: Ghi đè cấu hình mới vào template và xây dựng lại hệ thống
        if sampler_config:
            for key, value in vars(sampler_config).items():
                setattr(self.sampler_config, key, value)
                
        if pinn_config:
            for key, value in vars(pinn_config).items():
                setattr(self.pinn_config, key, value)
                
        if training_config:
            for key, value in vars(training_config).items():
                setattr(self.training_config, key, value)
                
        # Khởi tạo lại các thành phần con dựa trên cấu hình đã cập nhật
        self._build_system()

    def _build_system(self):
        # Khởi tạo Sampler
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.x_boundaries_tuple,
            y_boundaries_tuple=self.sampler_config.y_boundaries_tuple
        )
        
        # Tính toán các hằng số quy chuẩn
        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = 795774.715459
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        # Khởi tạo Mạng Neural (truyền kwargs từ SimpleNamespace)
        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **vars(self.pinn_config)
        )
        
        # Khởi tạo Động cơ PDE
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        # Khởi tạo Training Manager
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            **vars(self.training_config)
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
            epochs=epochs_adam, points_tensor=points_tensor,
            reluctivity_tensor=physical_properties_dictionary["reluctivity"],
            current_density_z_tensor=physical_properties_dictionary["current_density_z"],
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"],
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )
        
        print(f"--- L-BFGS Refinement ({epochs_lbfgs} Epochs) ---")
        self.training_manager_instance.train_lbfgs(
            epochs=epochs_lbfgs, points_tensor=points_tensor,
            reluctivity_tensor=physical_properties_dictionary["reluctivity"],
            current_density_z_tensor=physical_properties_dictionary["current_density_z"],
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"],
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor.requires_grad_(True)
        A_z_star = self.pinn_architecture_instance(points_tensor)
        A_z_phys = A_z_star * self.A0
        
        grad_A = torch.autograd.grad(
            outputs=A_z_phys, inputs=points_tensor,
            grad_outputs=torch.ones_like(A_z_phys), create_graph=False
        )[0]
        
        return A_z_phys.detach(), grad_A[:, 1:2].detach(), (-grad_A[:, 0:1]).detach()
"""

    # ---------------------------------------------------------
    # 2. CẬP NHẬT FILE TEST_SIMULATION.PY (U-Magnet & 512 Neurons)
    # ---------------------------------------------------------
    test_file = os.path.join(project_root, 'test_simulation.py')
    test_code = """import os
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
    # 1. KHỞI TẠO MÔ HÌNH VỚI TEMPLATE MẶC ĐỊNH
    model = ElectroMagneticPINN()
    
    # 2. ĐỊNH NGHĨA CÁC ĐỐI TƯỢNG CẤU HÌNH CẦN CẬP NHẬT
    new_sampler_config = SimpleNamespace(
        x_boundaries_tuple=(-0.08, 0.08),  # Mở rộng biên theo yêu cầu
        y_boundaries_tuple=(-0.08, 0.08)
    )
    
    new_pinn_config = SimpleNamespace(
        hidden_neurons=512,                # Tăng chiều rộng mạng lên 512 nơ-ron
        activation_function=nn.SiLU()      # Sử dụng SiLU
    )
    
    # Kích hoạt update để hệ thống tự động rebuild lại các class lõi
    model.update_configuration(
        sampler_config=new_sampler_config,
        pinn_config=new_pinn_config
    )
    
    # 3. THIẾT KẾ NAM CHÂM CHỮ U (Ghép từ 3 khối Segment để tạo mạch từ vòng)
    # Khối 1: Chân trái (Từ hóa hướng lên -> Cực Bắc)
    left_leg = Segment(outline=[[-0.04, -0.03], [-0.02, -0.03], [-0.02, 0.03], [-0.04, 0.03]])
    left_leg.set_material_properties(material="u_left", relative_permeability=1.05, coercive=[0.0, 800000.0])
    model.geometry_engine_instance.add_segment(left_leg)

    # Khối 2: Thanh đáy nối (Từ hóa hướng sang trái để đẩy mạch từ từ phải qua trái bên trong sắt)
    yoke = Segment(outline=[[-0.02, -0.03], [0.02, -0.03], [0.02, -0.01], [-0.02, -0.01]])
    yoke.set_material_properties(material="u_yoke", relative_permeability=1.05, coercive=[-800000.0, 0.0])
    model.geometry_engine_instance.add_segment(yoke)
    
    # Khối 3: Chân phải (Từ hóa hướng xuống -> Cực Nam)
    right_leg = Segment(outline=[[0.02, -0.03], [0.04, -0.03], [0.04, 0.03], [0.02, 0.03]])
    right_leg.set_material_properties(material="u_right", relative_permeability=1.05, coercive=[0.0, -800000.0])
    model.geometry_engine_instance.add_segment(right_leg)

    # 4. KÍCH HOẠT HÌNH ẢNH HỌC TỰ ĐỘNG THEO BIÊN MỚI
    xb = model.sampler_config.x_boundaries_tuple
    yb = model.sampler_config.y_boundaries_tuple
    model.geometry_engine_instance.plot_problem_definition(
        x_boundaries_tuple=xb, y_boundaries_tuple=yb, resolution=120
    )

    # 5. BẮT ĐẦU HUẤN LUYỆN (Tăng mạnh số điểm lấy mẫu theo yêu cầu)
    model.execute_training_process(
        number_of_uniform_points=15000,    # Tăng điểm phân bố đều
        number_of_interface_points=5000,   # Tăng điểm tại ranh giới
        distance_threshold=0.005,
        epochs_adam=4000,
        epochs_lbfgs=1000
    )
    
    # --- TRỰC QUAN HÓA KẾT QUẢ ---
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
    axs[0, 1].set_title("Magnetic Flux Density Magnitude ($\vert{}B\vert{}$)")
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
    
    print("Bắt đầu tái cấu trúc mô hình chữ U và hệ thống Cập nhật Cấu hình...")
    for file_path, content in files_to_update:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + "\n")
        print(f"[+] Đã ghi đè thành công: {os.path.basename(file_path)}")
        
    print("\nHOÀN TẤT! Bạn có thể chạy ngay `test_simulation.py` để chiêm ngưỡng nam châm chữ U với sức mạnh 512 neurons.")

if __name__ == "__main__":
    execute_refactor_u_magnet_and_templates()