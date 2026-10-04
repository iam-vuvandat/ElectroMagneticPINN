import os

def execute_final_refactoring():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # ---------------------------------------------------------
    # 1. ĐỒNG BỘ PINN_ARCHITECTURE.PY (Đưa tham số thành thuộc tính)
    # ---------------------------------------------------------
    pinn_file = os.path.join(project_root, 'pinn_architecture.py')
    pinn_code = """import torch
import torch.nn as nn

class PINNArchitecture(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=4, hidden_neurons=50, output_dim=1, domain_scale=0.05, activation_function=None):
        super().__init__()
        # ĐƯA RA LÀM THUỘC TÍNH
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.domain_scale = domain_scale
        
        if activation_function is None:
            self.activations = [nn.SiLU() for _ in range(self.hidden_layers)]
        elif isinstance(activation_function, list):
            self.activations = activation_function
        else:
            self.activations = [activation_function for _ in range(self.hidden_layers)]
            
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_neurons))
        layers.append(self.activations[0])
        
        for i in range(1, self.hidden_layers):
            layers.append(nn.Linear(self.hidden_neurons, self.hidden_neurons))
            layers.append(self.activations[i])
            
        layers.append(nn.Linear(self.hidden_neurons, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def boundary_factor(self, xy):
        x_factor = 1.0 - (xy[:, 0:1] / self.domain_scale)**2
        y_factor = 1.0 - (xy[:, 1:2] / self.domain_scale)**2
        return x_factor * y_factor

    def forward(self, xy):
        xy_normalized = xy / self.domain_scale
        raw_output = self.network(xy_normalized)
        return raw_output * self.boundary_factor(xy)
"""

    # ---------------------------------------------------------
    # 2. ĐỒNG BỘ TRAINING_MANAGER.PY (Lưu cấu hình và thêm Early Stopping)
    # ---------------------------------------------------------
    train_file = os.path.join(project_root, 'training_manager.py')
    train_code = """import torch
import torch.optim as optim

class TrainingManager:
    def __init__(
        self, 
        model, 
        pde_evaluator, 
        lr_adam=1e-3, 
        target_loss=0.0,
        lbfgs_lr=0.8, 
        lbfgs_max_iter=1000, 
        lbfgs_max_eval=1250,
        lbfgs_tolerance_grad=1e-8,
        lbfgs_tolerance_change=1e-10,
        lbfgs_history_size=50
    ):
        self.model = model
        self.pde_evaluator = pde_evaluator
        
        # ĐƯA RA LÀM THUỘC TÍNH
        self.base_lr_adam = lr_adam
        self.target_loss = target_loss
        self.lbfgs_lr = lbfgs_lr
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval
        self.lbfgs_tolerance_grad = lbfgs_tolerance_grad
        self.lbfgs_tolerance_change = lbfgs_tolerance_change
        self.lbfgs_history_size = lbfgs_history_size
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_lr_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_lr,
            max_iter=self.lbfgs_max_iter,
            max_eval=self.lbfgs_max_eval,
            tolerance_grad=self.lbfgs_tolerance_grad,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )

    def compute_loss(self, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        A_z_star = self.model(points_tensor)
        residual_star = self.pde_evaluator.compute_residual(
            xy=points_tensor, A_z_star=A_z_star, nu=reluctivity_tensor,
            J_z=current_density_z_tensor, H_cx=coercive_field_x_tensor, H_cy=coercive_field_y_tensor
        )
        return torch.mean(residual_star**2)

    def train_adam(self, epochs, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        self.model.train()
        best_loss = float('inf')
        best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
        
        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_lr_adam
            param_group['lr'] = self.base_lr_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(epochs):
            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor)
            
            if torch.isnan(loss) or loss.item() > 1.5 * best_loss:
                self.model.load_state_dict(best_model_state)
                for param_group in self.optimizer_adam.param_groups:
                    param_group['lr'] *= 0.8
                continue
                
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            current_loss_value = loss.item()
            if current_loss_value < best_loss:
                best_loss = current_loss_value
                best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                
            if self.target_loss > 0 and current_loss_value <= self.target_loss:
                print(f"Adam Epoch {epoch + 1}: Đạt ngưỡng target_loss. KẾT THÚC ADAM SỚM!")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Adam Epoch {epoch + 1}: Loss = {current_loss_value:.6e} | LR = {self.optimizer_adam.param_groups[0]['lr']:.3e}")

    def train_lbfgs(self, epochs, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        self.model.train()
        lbfgs_counter = [0]
        early_stop_triggered = False 
        
        def closure():
            nonlocal early_stop_triggered
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor)
            loss.backward(retain_graph=True)
            
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
                
            if self.target_loss > 0 and loss.item() <= self.target_loss:
                early_stop_triggered = True
            return loss
            
        for epoch in range(epochs):
            if early_stop_triggered:
                print(f"L-BFGS Epoch {epoch + 1}: Đạt ngưỡng target_loss. KẾT THÚC L-BFGS SỚM!")
                break
            self.optimizer_lbfgs.step(closure)
"""

    # ---------------------------------------------------------
    # 3. TÁI CẤU TRÚC LỚP BỌC ELECTRO_MAGNETIC_PINN (Dùng SimpleNamespace)
    # ---------------------------------------------------------
    emp_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    emp_code = """import torch
from types import SimpleNamespace
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from geometry_engine.geometry import Geometry
from physics_domain.collocation_sampler import CollocationSampler

class ElectroMagneticPINN:
    def __init__(
        self, 
        geometry_engine_configuration=None,
        collocation_sampler_configuration=None,
        pinn_architecture_configuration=None,
        training_manager_configuration=None
    ):
        # 1. LƯU TRỮ CẤU HÌNH BẰNG SIMPLENAMESPACE (Bảo vệ giá trị mặc định)
        self.geometry_engine_configuration = geometry_engine_configuration or SimpleNamespace()
        
        self.collocation_sampler_configuration = collocation_sampler_configuration or SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05)
        )
        
        self.pinn_architecture_configuration = pinn_architecture_configuration or SimpleNamespace()
        
        self.training_manager_configuration = training_manager_configuration or SimpleNamespace()
        
        # 2. KHỞI TẠO CÁC LỚP CON (LOẠI BỎ HOÀN TOÀN TIÊM PHỤ THUỘC)
        self.geometry_engine_instance = Geometry() # Geometry hiện không nhận args
        
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.collocation_sampler_configuration.x_boundaries_tuple,
            y_boundaries_tuple=self.collocation_sampler_configuration.y_boundaries_tuple
        )
        
        # Hằng số vật lý quy chuẩn
        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = 795774.715459
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        # 3. TRUYỀN CẤU HÌNH VÀO CÁC LỚP LÕI (Sử dụng vars() để bung Namespace thành kwargs)
        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **vars(self.pinn_architecture_configuration)
        )
        
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            **vars(self.training_manager_configuration)
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
    # 4. CẬP NHẬT FILE TEST_SIMULATION.PY (Giao diện người dùng tinh gọn)
    # ---------------------------------------------------------
    test_file = os.path.join(project_root, 'test_simulation.py')
    test_code = """import os
import sys
from types import SimpleNamespace

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
    # 1. ĐỊNH NGHĨA CÁC ĐỐI TƯỢNG CẤU HÌNH BẰNG SIMPLENAMESPACE (Gọn gàng & Chuyên nghiệp)
    sampler_config = SimpleNamespace(
        x_boundaries_tuple=(-0.05, 0.05),
        y_boundaries_tuple=(-0.05, 0.05)
    )
    
    pinn_config = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=64
    )
    
    training_config = SimpleNamespace(
        lr_adam=1e-3,
        target_loss=1e-3,
        lbfgs_lr=0.8,
        lbfgs_tolerance_grad=1e-8,
        lbfgs_tolerance_change=1e-10
    )

    # 2. KHỞI TẠO LỚP MẸ (Lớp mẹ sẽ tự xây dựng hệ thống con)
    model = ElectroMagneticPINN(
        collocation_sampler_configuration=sampler_config,
        pinn_architecture_configuration=pinn_config,
        training_manager_configuration=training_config
    )
    
    # 3. THÊM VẬT LIỆU BẰNG CÁCH GỌI THUỘC TÍNH TỪ CLASS MẸ
    top_magnet_vertices = [
        [-0.03, 0.015], [0.03, 0.015], [0.03, 0.025], [-0.03, 0.025]
    ]
    top_magnet = Segment(outline=top_magnet_vertices).set_material_properties(
        material="top_magnet", relative_permeability=1.05, coercive=[800000.0, 0.0]
    )
    model.geometry_engine_instance.add_segment(top_magnet)

    bottom_magnet_vertices = [
        [-0.03, -0.025], [0.03, -0.025], [0.03, -0.015], [-0.03, -0.015]
    ]
    bottom_magnet = Segment(outline=bottom_magnet_vertices).set_material_properties(
        material="bottom_magnet", relative_permeability=1.05, coercive=[-800000.0, 0.0]
    )
    model.geometry_engine_instance.add_segment(bottom_magnet)

    # 4. KÍCH HOẠT HÌNH ẢNH HỌC (Truy xuất thuộc tính bằng dot notation)
    xb = model.collocation_sampler_configuration.x_boundaries_tuple
    yb = model.collocation_sampler_configuration.y_boundaries_tuple
    model.geometry_engine_instance.plot_problem_definition(
        x_boundaries_tuple=xb, y_boundaries_tuple=yb, resolution=100
    )

    # 5. BẮT ĐẦU HUẤN LUYỆN
    model.execute_training_process(
        number_of_uniform_points=5000,
        number_of_interface_points=1500,
        distance_threshold=0.005,
        epochs_adam=4000,
        epochs_lbfgs=1000
    )
    
    # --- Trực quan hóa kết quả (Giữ nguyên) ---
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
        (pinn_file, pinn_code),
        (train_file, train_code),
        (emp_file, emp_code),
        (test_file, test_code)
    ]
    
    print("Bắt đầu đồng bộ và tái cấu trúc sử dụng SimpleNamespace...")
    for file_path, content in files_to_update:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + "\n")
        print(f"[+] Đã ghi đè và cấu hình lại: {os.path.basename(file_path)}")
        
    print("\nHOÀN TẤT RÀ SOÁT! Toàn bộ kiến trúc đã được đồng bộ, các tham số phơi bày rõ ràng và code test_simulation cực kỳ thanh lịch.")

if __name__ == "__main__":
    execute_final_refactoring()