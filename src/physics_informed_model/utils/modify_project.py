import os

def update_entire_project():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.abspath(os.path.join(base_dir, '..'))

    training_mgr_file = os.path.join(model_root, 'training_engine', 'training_manager.py')
    electro_pinn_file = os.path.join(model_root, 'electro_magnetic_pinn.py')
    visualizer_file = os.path.join(model_root, 'utils', 'training_visualizer.py')

    # 1. Cập nhật training_manager.py
    training_mgr_content = r'''import torch
import torch.optim as optim

class TrainingManager:
    def __init__(
        self, 
        model, 
        pde_evaluator, 
        learning_rate_adam=1e-3, 
        target_loss=0.0,
        lbfgs_learning_rate=0.8, 
        lbfgs_maximum_iterations=1000, 
        lbfgs_maximum_evaluations=1250,
        lbfgs_tolerance_gradient=1e-8,
        lbfgs_tolerance_change=1e-10,
        lbfgs_history_size=50,
        visualizer=None,
        visualizer_update_interval=100
    ):
        self.model = model
        self.pde_evaluator = pde_evaluator
        
        self.base_learning_rate_adam = learning_rate_adam
        self.target_loss = target_loss
        self.lbfgs_learning_rate = lbfgs_learning_rate
        self.lbfgs_maximum_iterations = lbfgs_maximum_iterations
        self.lbfgs_maximum_evaluations = lbfgs_maximum_evaluations
        self.lbfgs_tolerance_gradient = lbfgs_tolerance_gradient
        self.lbfgs_tolerance_change = lbfgs_tolerance_change
        self.lbfgs_history_size = lbfgs_history_size
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=self.lbfgs_maximum_iterations,
            max_eval=self.lbfgs_maximum_evaluations,
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )
        
        self.visualizer = visualizer
        self.visualizer_update_interval = visualizer_update_interval

    def compute_loss(self, domains_data):
        total_pde_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        total_interface_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for key, data in domains_data.items():
            domain = data["domain_object"]
            pts = data["interior_points"]
            
            if pts.shape[0] > 0:
                A_z_star = self.model.forward_network(key, pts)
                nu = domain.evaluate_reluctivity(pts)
                hx, hy = domain.evaluate_magnetization_vector(pts)
                jz = domain.evaluate_current_density(pts)
                
                residual_star = self.pde_evaluator.compute_residual(
                    xy=pts, A_z_star=A_z_star, nu=nu, J_z=jz, H_cx=hx, H_cy=hy
                )
                weighted_res = residual_star * domain.training_config.pde_weight
                pde_loss = torch.mean(weighted_res**2)
                total_pde_loss = total_pde_loss + pde_loss

            surf_pts = data["surface_points"]
            if surf_pts.shape[0] > 0 and key != 'background':
                Az_mat = self.model.forward_network(key, surf_pts)
                Az_bg = self.model.forward_network('background', surf_pts)
                
                continuity_loss = torch.mean((Az_mat - Az_bg)**2)
                total_interface_loss = total_interface_loss + domain.training_config.interface_weight * continuity_loss

        return total_pde_loss + total_interface_loss

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        self.model.train()
        
        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_learning_rate_adam
            param_group['lr'] = self.base_learning_rate_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=epochs, eta_min=1e-6)
        stop_reason = f"Hoàn thành toàn bộ {epochs} epochs."
        
        for epoch in range(epochs):
            if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
                domains_data = resample_fn()

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            
            if torch.isnan(loss):
                print(f"Adam Epoch {epoch + 1}: CẢNH BÁO - Loss bị NaN. Bỏ qua step này.")
                continue
                
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            current_loss_value = loss.item()
            
            if self.target_loss > 0 and current_loss_value <= self.target_loss:
                stop_reason = f"Đạt ngưỡng target_loss ({self.target_loss}) tại epoch {epoch + 1}."
                print(f"Adam Epoch {epoch + 1}: {stop_reason} KẾT THÚC ADAM SỚM!")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Adam Epoch {epoch + 1}: Loss = {current_loss_value:.6e} | LR = {self.optimizer_adam.param_groups[0]['lr']:.3e}")
                
            if self.visualizer and (epoch + 1) % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
                
        print(f"[*] KẾT THÚC HUẤN LUYỆN ADAM. Lý do: {stop_reason}")

    def train_lbfgs(self, epochs, domains_data):
        self.model.train()
        lbfgs_counter = [0]
        early_stop_triggered = False 
        
        def closure():
            nonlocal early_stop_triggered
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            loss.backward(retain_graph=True)
            
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
                
            if self.visualizer and lbfgs_counter[0] % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(50000 + lbfgs_counter[0], loss.item(), domains_data)
                
            if self.target_loss > 0 and loss.item() <= self.target_loss:
                early_stop_triggered = True
            return loss
            
        for epoch in range(epochs):
            if early_stop_triggered:
                print(f"L-BFGS Epoch {epoch + 1}: Đạt ngưỡng target_loss. KẾT THÚC L-BFGS SỚM!")
                break
            self.optimizer_lbfgs.step(closure)
'''
    with open(training_mgr_file, 'w', encoding='utf-8') as f:
        f.write(training_mgr_content)

    # 2. Cập nhật electro_magnetic_pinn.py
    electro_pinn_content = r'''import torch
import torch.nn as nn
from architectures.xpinn_architecture import XPINNArchitecture
from types import SimpleNamespace
from training_engine.training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from simulation_domain_engine.simulation_domain import SimulationDomain
from physics_domain.collocation_sampler import CollocationSampler
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY
from utils.training_visualizer import TrainingVisualizer

class ElectroMagneticPINN:
    def __init__(self):
        self.sampler_config = SimpleNamespace(
            dynamic_resampling=True,
            resample_interval=100
        )
        
        self.training_config = SimpleNamespace(
            learning_rate_adam=1e-3,
            target_loss=1e-3,
            epochs_adam=4000,
            epochs_lbfgs=1000,
            lbfgs_learning_rate=0.8,
            lbfgs_maximum_iterations=1000,
            lbfgs_maximum_evaluations=1250,
            lbfgs_tolerance_gradient=1e-8,
            lbfgs_tolerance_change=1e-10,
            lbfgs_history_size=50
        )
        
        self.visualization_config = SimpleNamespace(
            active=False,
            update_interval=100,
            output_directory="animation_frames",
            resolution=100,
            gif_filename="training_process.gif",
            gif_fps=10
        )
        
        self.physics_config = SimpleNamespace(
            scale_L0=0.05,
            scale_H0=1200000.0,
            scale_nu0=VACUUM_RELUCTIVITY
        )
        
        self.simulation_domain_engine_instance = SimulationDomain()
        self.pinn_architecture_instance = None
        self.training_manager_instance = None
        self.maxwell_pde_loss_instance = None
        self.collocation_sampler_instance = None
        self.visualizer_instance = None

    def update_electromagnetic_pinn(self):
        self._build_system()

    def _build_system(self):
        self.computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Hệ thống vật lý được khởi tạo trên: {self.computation_device}")
        
        self.collocation_sampler_instance = CollocationSampler(device=self.computation_device)
        
        self.L0 = self.physics_config.scale_L0
        self.H0 = self.physics_config.scale_H0
        self.nu0 = self.physics_config.scale_nu0
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        self.pinn_architecture_instance = XPINNArchitecture(
            electro_magnetic_pinn=self,
            domain_scale=self.L0
        ).to(self.computation_device)
        
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        self.visualizer_instance = None
        if self.visualization_config.active:
            self.visualizer_instance = TrainingVisualizer(parent_pinn=self)
            
        manager_kwargs = vars(self.training_config).copy()
        manager_kwargs.pop('epochs_adam', None)
        manager_kwargs.pop('epochs_lbfgs', None)
            
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            visualizer=self.visualizer_instance,
            visualizer_update_interval=self.visualization_config.update_interval,
            **manager_kwargs
        )

    def execute_training_process(self):
        if self.pinn_architecture_instance is None:
            self._build_system()

        def resample_data_fn():
            return self.collocation_sampler_instance.generate_all_domain_points(
                simulation_domain_object=self.simulation_domain_engine_instance
            )

        domains_data = resample_data_fn()
        
        print(f"--- Standard Adam Training ({self.training_config.epochs_adam} Epochs) ---")
        self.training_manager_instance.train_adam(
            epochs=self.training_config.epochs_adam, 
            domains_data=domains_data,
            dynamic_resampling=self.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn,
            resample_interval=self.sampler_config.resample_interval
        )
        
        print(f"--- L-BFGS Refinement ({self.training_config.epochs_lbfgs} Epochs) ---")
        self.training_manager_instance.train_lbfgs(
            epochs=self.training_config.epochs_lbfgs, 
            domains_data=domains_data
        )
        
        if self.visualizer_instance:
            print("Đang xuất file GIF Animation...")
            self.visualizer_instance.generate_gif()

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor = points_tensor.to(self.computation_device).clone().requires_grad_(True)
        
        props = self.simulation_domain_engine_instance.evaluate_global_physical_properties(points_tensor)
        mat_id = props["material_classification"]
        
        A_z_star_total = torch.zeros((points_tensor.shape[0], 1), device=self.computation_device)
        
        for domain in self.simulation_domain_engine_instance.domains_list:
            d_id = domain.domain_id
            if getattr(domain, 'domain_name', '') == 'background':
                net_key = 'background'
            else:
                net_key = f'domain_{d_id}'
                
            mask = (mat_id == float(d_id)).float()
            if mask.any():
                Az_sub = self.pinn_architecture_instance.forward_network(net_key, points_tensor)
                A_z_star_total = A_z_star_total + mask * Az_sub

        A_z_phys = A_z_star_total * self.A0
        
        grad_A = torch.autograd.grad(
            outputs=A_z_phys, inputs=points_tensor,
            grad_outputs=torch.ones_like(A_z_phys), create_graph=False
        )[0]
        
        return A_z_phys.detach(), grad_A[:, 1:2].detach(), (-grad_A[:, 0:1]).detach()
'''
    with open(electro_pinn_file, 'w', encoding='utf-8') as f:
        f.write(electro_pinn_content)

    # 3. Cập nhật training_visualizer.py
    visualizer_content = r'''import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio

class TrainingVisualizer:
    def __init__(self, parent_pinn):
        self.parent_pinn = parent_pinn
        self.output_directory = self.parent_pinn.visualization_config.output_directory
        self.x_bounds = self.parent_pinn.simulation_domain_engine_instance.x_boundaries_tuple
        self.y_bounds = self.parent_pinn.simulation_domain_engine_instance.y_boundaries_tuple
        self.resolution = self.parent_pinn.visualization_config.resolution
        
        self.frame_paths = []
        self.loss_history = []
        self.epoch_history = []
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        x_coords = np.linspace(self.x_bounds[0], self.x_bounds[1], self.resolution)
        y_coords = np.linspace(self.y_bounds[0], self.y_bounds[1], self.resolution)
        self.X_grid, self.Y_grid = np.meshgrid(x_coords, y_coords)
        self.eval_points_tensor = torch.tensor(
            np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel())), 
            dtype=torch.float32
        )

    def save_frame(self, epoch, loss_value, domains_data):
        self.loss_history.append(loss_value)
        self.epoch_history.append(epoch)
        
        model = self.parent_pinn.pinn_architecture_instance
        pde_evaluator = self.parent_pinn.maxwell_pde_loss_instance
        sim_domain = self.parent_pinn.simulation_domain_engine_instance
        
        model.eval()
        computation_device = next(model.parameters()).device
        eval_points = self.eval_points_tensor.to(computation_device).clone().requires_grad_(True)
        
        phys_props = sim_domain.evaluate_global_physical_properties(eval_points)
        nu_grid = phys_props["reluctivity"]
        jz_grid = phys_props["current_density_z"]
        hcx_grid = phys_props["coercive_field_x"]
        hcy_grid = phys_props["coercive_field_y"]
        
        A_z_phys, B_x_phys, B_y_phys = self.parent_pinn.evaluate_fields(eval_points)
        A_z_star = A_z_phys / self.parent_pinn.A0
        
        residual_pred = pde_evaluator.compute_residual(eval_points, A_z_star, nu_grid, jz_grid, hcx_grid, hcy_grid)
        
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy**2 + B_y_numpy**2)
        residual_numpy = residual_pred.detach().cpu().numpy().reshape(self.resolution, self.resolution)
        
        all_pts = []
        for key, d in domains_data.items():
            if d["interior_points"].shape[0] > 0:
                all_pts.append(d["interior_points"].detach().cpu().numpy())
            if d["surface_points"].shape[0] > 0:
                all_pts.append(d["surface_points"].detach().cpu().numpy())
        points_numpy = np.vstack(all_pts) if all_pts else np.empty((0, 2))
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        if points_numpy.shape[0] > 0:
            axs[0, 0].scatter(points_numpy[:, 0], points_numpy[:, 1], s=1, c='black', alpha=0.5)
        axs[0, 0].set_title("Collocation Points")
        axs[0, 0].set_xlim(self.x_bounds)
        axs[0, 0].set_ylim(self.y_bounds)
        axs[0, 0].set_aspect('equal')
        
        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, np.abs(residual_numpy), levels=50, cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=1e1))
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')
        
        axs[0, 2].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 2].set_yscale('log')
        axs[0, 2].set_title("Loss Optimization")
        axs[0, 2].set_xlabel("Epoch")
        axs[0, 2].set_ylabel("Loss")
        axs[0, 2].grid(True, which="both", ls="-", alpha=0.2)
        
        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=50, cmap="jet")
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        contour_b = axs[1, 1].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=50, cmap="rainbow")
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title("Flux Density Magnitude ($|B|$)")
        axs[1, 1].set_aspect('equal')

        contour_b_bg = axs[1, 2].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=50, cmap="rainbow", alpha=0.4)
        fig.colorbar(contour_b_bg, ax=axs[1, 2], label="|B| (T)")
        
        step = max(1, self.resolution // 20)
        axs[1, 2].quiver(self.X_grid[::step, ::step], self.Y_grid[::step, ::step], 
                         B_x_numpy[::step, ::step], B_y_numpy[::step, ::step], 
                         color='black', pivot='mid')
        axs[1, 2].set_title("Flux Density Vectors (B)")
        axs[1, 2].set_aspect('equal')
        
        fig.suptitle(f"PINN Training Process - Epoch {epoch}", fontsize=16)
        plt.tight_layout()
        
        frame_name = os.path.join(self.output_directory, f"frame_{epoch:05d}.png")
        plt.savefig(frame_name, dpi=100)
        plt.close(fig)
        
        self.frame_paths.append(frame_name)
        model.train()

    def generate_gif(self):
        output_filename = self.parent_pinn.visualization_config.gif_filename
        fps = self.parent_pinn.visualization_config.gif_fps
        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")
'''
    with open(visualizer_file, 'w', encoding='utf-8') as f:
        f.write(visualizer_content)

if __name__ == "__main__":
    update_entire_project()