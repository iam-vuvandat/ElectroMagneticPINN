import os

def refactor_hyperparameters_to_attributes():
    base_directory = os.path.dirname(os.path.abspath(__file__))
    project_root_directory = os.path.abspath(os.path.join(base_directory, '..'))
    
    training_manager_path = os.path.join(project_root_directory, 'training_manager.py')
    electro_magnetic_path = os.path.join(project_root_directory, 'electro_magnetic_pinn.py')
    test_simulation_path = os.path.join(project_root_directory, 'test_simulation.py')

    training_manager_code = """import torch
import torch.optim as optim

class TrainingManager:
    def __init__(self, model, pde_evaluator, lr_adam=1e-3, lbfgs_lr=0.8, lbfgs_max_iter=1000, lbfgs_max_eval=1250):
        self.model = model
        self.pde_evaluator = pde_evaluator
        self.base_lr_adam = lr_adam
        
        self.lbfgs_lr = lbfgs_lr
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_lr_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_lr,
            max_iter=self.lbfgs_max_iter,
            max_eval=self.lbfgs_max_eval,
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

    def compute_loss(self, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        A_z_star = self.model(points_tensor)
        
        residual_star = self.pde_evaluator.compute_residual(
            xy=points_tensor,
            A_z_star=A_z_star,
            nu=reluctivity_tensor,
            J_z=current_density_z_tensor,
            H_cx=coercive_field_x_tensor,
            H_cy=coercive_field_y_tensor
        )
        
        loss_pde = torch.mean(residual_star**2)
        return loss_pde

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
            
            loss = self.compute_loss(
                points_tensor, reluctivity_tensor, 
                current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor
            )
            
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
            
            if (epoch + 1) % 100 == 0:
                current_lr = self.optimizer_adam.param_groups[0]['lr']
                print(f"Adam Epoch {epoch + 1}: Loss = {current_loss_value:.6e} | LR = {current_lr:.3e}")

    def train_lbfgs(self, epochs, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        self.model.train()
        lbfgs_counter = [0]
        
        def closure():
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(
                points_tensor, reluctivity_tensor, 
                current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor
            )
            loss.backward(retain_graph=True)
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
            return loss
            
        self.optimizer_lbfgs.step(closure)
"""

    electro_magnetic_code = """import torch
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss

class ElectroMagneticPINN:
    def __init__(self, geometry_engine_instance, collocation_sampler_instance, lr_adam=1e-3, lbfgs_lr=0.8, lbfgs_max_iter=1000, lbfgs_max_eval=1250):
        self.geometry_engine_instance = geometry_engine_instance
        self.collocation_sampler_instance = collocation_sampler_instance
        
        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = self.geometry_engine_instance.vacuum_reluctivity
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        self.pinn_architecture_instance = PINNArchitecture(domain_scale=self.L0)
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        self.lr_adam = lr_adam
        self.lbfgs_lr = lbfgs_lr
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval
        
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            lr_adam=self.lr_adam,
            lbfgs_lr=self.lbfgs_lr,
            lbfgs_max_iter=self.lbfgs_max_iter,
            lbfgs_max_eval=self.lbfgs_max_eval
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

    test_simulation_code = """import os
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
        [-0.03, 0.015],
        [0.03, 0.015],
        [0.03, 0.025],
        [-0.03, 0.025]
    ]
    top_magnet = Segment(top_magnet_vertices).set_material_properties(
        name="top_magnet",
        relative_permeability=1.05,
        coercive_field_x=800000.0,
        coercive_field_y=0.0
    )
    geometry_instance.add_segment(top_magnet)

    bottom_magnet_vertices = [
        [-0.03, -0.025],
        [0.03, -0.025],
        [0.03, -0.015],
        [-0.03, -0.015]
    ]
    bottom_magnet = Segment(bottom_magnet_vertices).set_material_properties(
        name="bottom_magnet",
        relative_permeability=1.05,
        coercive_field_x=-800000.0,
        coercive_field_y=0.0
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
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].set_aspect('equal')
    
    contour_b = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=60, cmap="rainbow")
    fig1.colorbar(contour_b, ax=axs[0, 1], label="|B| (T)")
    axs[0, 1].set_title("Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_xlabel("x (m)")
    axs[0, 1].set_ylabel("y (m)")
    axs[0, 1].set_aspect('equal')
    
    contour_bx = axs[1, 0].contourf(X_grid, Y_grid, B_x_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_bx, ax=axs[1, 0], label="B_x (T)")
    axs[1, 0].set_title("Magnetic Field Component ($B_x$)")
    axs[1, 0].set_xlabel("x (m)")
    axs[1, 0].set_ylabel("y (m)")
    axs[1, 0].set_aspect('equal')
    
    contour_by = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_by, ax=axs[1, 1], label="B_y (T)")
    axs[1, 1].set_title("Magnetic Field Component ($B_y$)")
    axs[1, 1].set_xlabel("x (m)")
    axs[1, 1].set_ylabel("y (m)")
    axs[1, 1].set_aspect('equal')
    
    fig1.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    contour_b_bg = ax2.contourf(X_grid, Y_grid, B_mag_grid, levels=60, cmap="rainbow", alpha=0.4)
    fig2.colorbar(contour_b_bg, ax=ax2, label="|B| (T)")
    
    step = 4
    ax2.quiver(X_grid[::step, ::step], Y_grid[::step, ::step], B_x_grid[::step, ::step], B_y_grid[::step, ::step], color='black', pivot='mid')
    ax2.set_title("Magnetic Flux Density Vectors (B)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect('equal')
    
    fig2.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
"""

    with open(training_manager_path, 'w', encoding='utf-8') as f:
        f.write(training_manager_code)
        
    with open(electro_magnetic_path, 'w', encoding='utf-8') as f:
        f.write(electro_magnetic_code)
        
    with open(test_simulation_path, 'w', encoding='utf-8') as f:
        f.write(test_simulation_code)
        
    print("Đã tái cấu trúc: Các siêu tham số L-BFGS giờ đây đã trở thành thuộc tính của lớp và có thể điều khiển trực tiếp từ test_simulation.py.")

if __name__ == '__main__':
    refactor_hyperparameters_to_attributes()