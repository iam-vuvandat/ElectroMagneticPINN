import os

def apply_all_fixes_and_configurations():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    maxwell_pde_file = os.path.join(project_root, 'physics_domain', 'physical_equations', 'maxwell_pde_loss.py')
    training_manager_file = os.path.join(project_root, 'training_engine', 'training_manager.py')
    pinn_core_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    test_sim_file = os.path.join(project_root, 'tests', 'test_simulation.py')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: maxwell_pde_loss.py
    # Sửa triệt để lỗi typo xdAz_dy_star -> dAz_dy_star
    # =========================================================================
    files_to_update[maxwell_pde_file] = '''import torch

class MaxwellPDELoss:
    def __init__(self, electro_magnetic_pinn=None, L0=0.1, H0=800000.0, nu0=795774.715459):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        self._L0 = L0
        self._H0 = H0
        self._nu0 = nu0

    @property
    def L0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.L0
        return self._L0

    @property
    def H0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.H0
        return self._H0

    @property
    def nu0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.nu0
        return self._nu0

    def compute_linear_pde_residual(self, xy, A_z_star, nu_phys, J_z, H_cx, H_cy, nu_star_net=None):
        if nu_star_net is not None:
            nu_star = nu_star_net
        else:
            nu_star = nu_phys / self.nu0

        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)

        # 1. Gradient của Az* -> B*
        grad_A_star = torch.autograd.grad(
            outputs=A_z_star,
            inputs=xy,
            grad_outputs=torch.ones_like(A_z_star),
            create_graph=True
        )[0]

        dAz_dx_star = grad_A_star[:, 0:1] * self.L0
        dAz_dy_star = grad_A_star[:, 1:2] * self.L0

        # 2. H* = nu* * B* - Hc*
        Hx_star = nu_star * dAz_dy_star - H_cx_star
        Hy_star = -nu_star * dAz_dx_star - H_cy_star

        # 3. curl(H*) = dHy*/dx* - dHx*/dy*
        grad_Hx = torch.autograd.grad(outputs=Hx_star, inputs=xy, grad_outputs=torch.ones_like(Hx_star), create_graph=True)[0]
        grad_Hy = torch.autograd.grad(outputs=Hy_star, inputs=xy, grad_outputs=torch.ones_like(Hy_star), create_graph=True)[0]

        dHy_dx_star = grad_Hy[:, 0:1] * self.L0
        dHx_dy_star = grad_Hx[:, 1:2] * self.L0

        res_pde = dHy_dx_star - dHx_dy_star - J_z_star
        nu_scale = torch.clamp(nu_star.detach(), min=1e-6)
        return res_pde / nu_scale

    def compute_fading_weight(self, domain, xy):
        beta = float(getattr(domain.training_config, 'pde_fading_beta', 80.0))
        if beta <= 0.0:
            return torch.ones((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)

        if self.electro_magnetic_pinn is not None:
            sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
            mat_verts = [
                d.outline_tensor.to(device=xy.device, dtype=xy.dtype)
                for d in sim_domain.domains_list
                if getattr(d, 'domain_name', '') != 'background' and d.outline_tensor is not None
            ]
            if mat_verts:
                all_verts = torch.cat(mat_verts, dim=0)
                dists = torch.cdist(xy.detach(), all_verts, p=2.0)
                min_corner_dist, _ = torch.min(dists, dim=1, keepdim=True)
                corner_dist_star = min_corner_dist / self.L0
                return 1.0 - torch.exp(-beta * corner_dist_star)

        sdf = domain.compute_signed_distance_field(xy).view(-1, 1)
        abs_sdf_star = torch.abs(sdf.detach()) / self.L0
        return 1.0 - torch.exp(-beta * abs_sdf_star)
'''

    # =========================================================================
    # 2. CẬP NHẬT: electro_magnetic_pinn.py
    # Bổ sung các thuộc tính cấu hình mặc định vào self.training_config
    # =========================================================================
    files_to_update[pinn_core_file] = '''import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from types import SimpleNamespace
from architectures.xpinn_architecture import XPINNArchitecture
from training_engine.training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from physics_domain.physical_equations.interface_loss import InterfaceLoss
from simulation_domain_engine.simulation_domain import SimulationDomain
from physics_domain.collocation_sampler import CollocationSampler
from utils.training_visualizer import TrainingVisualizer

class ElectroMagneticPINN:
    def __init__(self):
        self.sampler_config = SimpleNamespace(
            dynamic_resampling=True,
            resample_interval=100,
            resample_points_before_lbfgs_stage=True
        )
        
        self.training_config = SimpleNamespace(
            enable_linear_warmup_stage=True,
            enable_nonlinear_adaptation_stage=True,
            enable_lbfgs_refinement_stage=True,
            linear_warmup_epochs_ratio=0.3,

            adam_learning_rate=1e-3,
            number_of_adam_epochs=2500,
            target_loss_threshold=1e-5,

            # Cấu hình FPI Picard Outer Loop
            fpi_outer_iterations=10,
            fpi_alpha_damping=0.25,
            fpi_tolerance=1e-5,

            # Cấu hình Pre-train và Fit Net_nu (Đã linh hoạt hóa)
            pretrain_nu_epochs=300,
            pretrain_nu_lr=1e-2,
            fit_nu_epochs=250,
            fit_nu_lr=3e-3,

            # Cấu hình kiểm soát Gradient và Scheduler
            grad_clip_max_norm=1.0,
            adam_eta_min=1e-6,
            max_non_finite_streak=5,
            adam_log_interval=100,
            lbfgs_log_interval=25,

            # Cấu hình L-BFGS Refinement
            lbfgs_learning_rate=0.5,
            number_of_lbfgs_iterations=1500,
            lbfgs_maximum_evaluations=2500,
            lbfgs_tolerance_gradient=1e-12,
            lbfgs_tolerance_change=1e-12,
            lbfgs_history_size=50
        )
        
        self.visualization_config = SimpleNamespace(
            active=False,
            update_interval=200,
            output_directory="animation_frames",
            resolution=80,
            generate_gif_animation=True,
            gif_filename="training_process.gif",
            generate_mp4_video=True,
            mp4_filename="training_process.mp4",
            animation_fps=5,
            gif_fps=5,
            mp4_codec="libx264",
            mp4_quality=8
        )
        
        self.simulation_domain_engine_instance = SimulationDomain(electro_magnetic_pinn=self)
        self.pinn_architecture_instance = None
        self.training_manager_instance = None
        self.maxwell_pde_loss_instance = None
        self.interface_loss_instance = None
        self.collocation_sampler_instance = None
        self.visualizer_instance = None

    @property
    def physics_config(self):
        return self.simulation_domain_engine_instance.physics_config

    @property
    def L0(self):
        return self.simulation_domain_engine_instance.L0

    @property
    def H0(self):
        return self.simulation_domain_engine_instance.H0

    @property
    def nu0(self):
        return self.simulation_domain_engine_instance.nu0

    @property
    def A0(self):
        return self.simulation_domain_engine_instance.A0

    def update_electromagnetic_pinn(self):
        self._build_system()

    def _build_system(self):
        self.simulation_domain_engine_instance.validate_non_overlapping_domains()
        self.computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Hệ thống vật lý được khởi tạo trên: {self.computation_device}")
        
        self.collocation_sampler_instance = CollocationSampler(
            electro_magnetic_pinn=self,
            device=self.computation_device
        )
        
        self.pinn_architecture_instance = XPINNArchitecture(
            electro_magnetic_pinn=self
        ).to(self.computation_device)
        
        self.maxwell_pde_loss_instance = MaxwellPDELoss(electro_magnetic_pinn=self)
        self.interface_loss_instance = InterfaceLoss(electro_magnetic_pinn=self)
        
        self.visualizer_instance = None
        if self.visualization_config.active:
            self.visualizer_instance = TrainingVisualizer(electro_magnetic_pinn=self)
            
        self.training_manager_instance = TrainingManager(electro_magnetic_pinn=self)

        # Pre-train Net_nu theo cấu hình linh hoạt
        domains_data = self.collocation_sampler_instance.generate_all_domain_points()
        pre_ep = getattr(self.training_config, 'pretrain_nu_epochs', 300)
        pre_lr = getattr(self.training_config, 'pretrain_nu_lr', 1e-2)
        self.training_manager_instance.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)

    def evaluate_fields(self, points_tensor, return_numpy=False):
        pinn_arch = self.pinn_architecture_instance
        sim_domain = self.simulation_domain_engine_instance
        device = points_tensor.device

        phys_props = sim_domain.evaluate_global_physical_properties(points_tensor)
        mat_id = phys_props["material_classification"]

        A_z_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
        B_x_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
        B_y_phys = torch.zeros((points_tensor.shape[0], 1), device=device)

        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{d_id}'
            mask_bool = (mat_id == float(d_id))
            
            if mask_bool.any():
                Az_star = pinn_arch.forward_network(net_key, points_tensor)
                grad_star = torch.autograd.grad(
                    outputs=Az_star, inputs=points_tensor,
                    grad_outputs=torch.ones_like(Az_star), create_graph=False
                )[0]

                Az_phys = Az_star * self.A0
                Bx_phys = grad_star[:, 1:2] * self.A0
                By_phys = -grad_star[:, 0:1] * self.A0

                A_z_phys = torch.where(mask_bool, Az_phys, A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_phys, B_x_phys)
                B_y_phys = torch.where(mask_bool, By_phys, B_y_phys)

        if return_numpy:
            return (
                A_z_phys.detach().cpu().numpy(),
                B_x_phys.detach().cpu().numpy(),
                B_y_phys.detach().cpu().numpy()
            )

        return A_z_phys.detach(), B_x_phys.detach(), B_y_phys.detach()

    def plot_fields(self, resolution=180, save_path=None, show=False, figsize=(12, 10)):
        sim_domain = self.simulation_domain_engine_instance
        xb, yb = sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple
        x_coords = np.linspace(xb[0], xb[1], resolution)
        y_coords = np.linspace(yb[0], yb[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        
        device = self.computation_device
        eval_pts = torch.tensor(
            np.column_stack((X_grid.ravel(), Y_grid.ravel())),
            dtype=torch.float32, device=device
        ).requires_grad_(True)
        
        A_z_np, B_x_np, B_y_np = self.evaluate_fields(eval_pts, return_numpy=True)

        A_z_grid = A_z_np.reshape(resolution, resolution)
        B_x_grid = B_x_np.reshape(resolution, resolution)
        B_y_grid = B_y_np.reshape(resolution, resolution)
        B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)

        props = sim_domain.evaluate_global_physical_properties(eval_pts)
        mat_ids = props["material_classification"].squeeze()
        
        mu_r_tensor = torch.ones_like(mat_ids, device=device)
        for domain in sim_domain.domains_list:
            if getattr(domain, 'domain_name', '') != 'background':
                mask = (mat_ids == float(domain.domain_id))
                net_k = f'domain_{domain.domain_id}'
                if hasattr(self.pinn_architecture_instance, 'nu_networks') and net_k in self.pinn_architecture_instance.nu_networks:
                    nu_star_p = self.pinn_architecture_instance.forward_nu_network(net_k, eval_pts)
                    mu_val = (1.0 / torch.clamp(nu_star_p.squeeze(), min=1e-6))
                    mu_r_tensor[mask] = mu_val[mask]
                else:
                    mu_val = float(getattr(domain.physics_config, 'relative_permeability', 1.0))
                    mu_r_tensor[mask] = mu_val

        mu_r_grid = mu_r_tensor.detach().cpu().numpy().reshape(resolution, resolution)

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        c0 = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=60, cmap="jet")
        fig.colorbar(c0, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
        axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
        axs[0, 0].set_aspect('equal')

        b_max = max(float(np.max(B_mag_grid)), 1e-3)
        c1 = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=np.linspace(0, b_max, 60), cmap="jet")
        fig.colorbar(c1, ax=axs[0, 1], label=r"$|B|$ (T)")
        axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
        axs[0, 1].set_aspect('equal')

        c2 = axs[1, 0].contourf(X_grid, Y_grid, mu_r_grid, levels=60, cmap="cividis")
        fig.colorbar(c2, ax=axs[1, 0], label=r"$\mu_r$")
        axs[1, 0].set_title(r"Relative Permeability Distribution ($\mu_r$)")
        axs[1, 0].set_aspect('equal')

        c3 = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
        fig.colorbar(c3, ax=axs[1, 1], label=r"$B_y$ (T)")
        axs[1, 1].set_title(r"Field Component ($B_y$)")
        axs[1, 1].set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axs

    def execute_training_process(self):
        if self.pinn_architecture_instance is None:
            self._build_system()

        def resample_data_fn():
            return self.collocation_sampler_instance.generate_all_domain_points()

        domains_data = resample_data_fn()
        t_cfg = self.training_config

        # 1. Pre-training Net_nu
        pre_ep = getattr(t_cfg, 'pretrain_nu_epochs', 300)
        pre_lr = getattr(t_cfg, 'pretrain_nu_lr', 1e-2)
        self.training_manager_instance.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)

        # 2. Adam Stage
        total_adam_epochs = t_cfg.number_of_adam_epochs
        warmup_ratio = float(getattr(t_cfg, 'linear_warmup_epochs_ratio', 0.3))
        epochs_warmup = int(total_adam_epochs * warmup_ratio) if getattr(t_cfg, 'enable_linear_warmup_stage', True) else 0
        epochs_adaptation = (total_adam_epochs - epochs_warmup) if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) else 0

        if getattr(t_cfg, 'enable_linear_warmup_stage', True) and epochs_warmup > 0:
            print(f"\\n--- [STAGE 1A] Adam Linear Warm-up ({epochs_warmup} Epochs) ---")
            domains_data = self.training_manager_instance.train_adam(
                epochs=epochs_warmup, domains_data=domains_data,
                dynamic_resampling=self.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn, resample_interval=self.sampler_config.resample_interval
            )

        if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) and epochs_adaptation > 0:
            fpi_iters = int(getattr(t_cfg, 'fpi_outer_iterations', 10))
            alpha_damp = float(getattr(t_cfg, 'fpi_alpha_damping', 0.25))
            fpi_tol = float(getattr(t_cfg, 'fpi_tolerance', 1e-5))
            epochs_per_fpi = max(1, epochs_adaptation // fpi_iters)

            fit_ep = getattr(t_cfg, 'fit_nu_epochs', 250)
            fit_lr = getattr(t_cfg, 'fit_nu_lr', 3e-3)

            print(f"\\n--- [STAGE 1B] FPI Outer Loop ({fpi_iters} iterations, {epochs_per_fpi} epochs/iter) ---")
            for f_iter in range(fpi_iters):
                print(f"\\n>>> [FPI OUTER LOOP {f_iter + 1}/{fpi_iters}] <<<")
                domains_data = self.training_manager_instance.train_adam(
                    epochs=epochs_per_fpi, domains_data=domains_data,
                    dynamic_resampling=self.sampler_config.dynamic_resampling,
                    resample_fn=resample_data_fn, resample_interval=self.sampler_config.resample_interval
                )
                max_diff = self.training_manager_instance.fit_nu_network_step(
                    domains_data, alpha_damping=alpha_damp, epochs=fit_ep, lr=fit_lr
                )
                print(f"[*] Outer FPI Iter {f_iter + 1}: Max Nu Diff = {max_diff:.4e}")
                if max_diff < fpi_tol:
                    print(f"[✔] FPI Hội tụ sớm tại vòng {f_iter + 1}!")
                    break

        # 3. L-BFGS Stage
        if getattr(t_cfg, 'enable_lbfgs_refinement_stage', True) and t_cfg.number_of_lbfgs_iterations > 0:
            if getattr(self.sampler_config, 'resample_points_before_lbfgs_stage', True):
                domains_data = resample_data_fn()

            lbfgs_iters = int(t_cfg.number_of_lbfgs_iterations)
            print(f"\\n--- [STAGE 2] L-BFGS Refinement ({lbfgs_iters} Iterations) ---")
            self.training_manager_instance.train_lbfgs(max_iterations=lbfgs_iters, domains_data=domains_data)

        if self.visualizer_instance:
            self.visualizer_instance.generate_animations()
'''

    # =========================================================================
    # 3. CẬP NHẬT: training_manager.py
    # Đưa các tham số huấn luyện phụ thuộc hoàn toàn vào training_config
    # =========================================================================
    files_to_update[training_manager_file] = '''import time
import torch
import torch.optim as optim

class TrainingManager:
    def __init__(self, electro_magnetic_pinn):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        
        t_cfg = self.electro_magnetic_pinn.training_config
        
        self.base_learning_rate_adam = getattr(t_cfg, 'adam_learning_rate', getattr(t_cfg, 'learning_rate_adam', 1e-3))
        self.target_loss_threshold = getattr(t_cfg, 'target_loss_threshold', getattr(t_cfg, 'target_loss', 1e-5))
        self.lbfgs_learning_rate = getattr(t_cfg, 'lbfgs_learning_rate', 0.5)
        self.lbfgs_maximum_iterations = getattr(t_cfg, 'number_of_lbfgs_iterations', getattr(t_cfg, 'lbfgs_maximum_iterations', 1500))
        self.lbfgs_maximum_evaluations = getattr(t_cfg, 'lbfgs_maximum_evaluations', 2500)
        self.lbfgs_tolerance_gradient = getattr(t_cfg, 'lbfgs_tolerance_gradient', 1e-12)
        self.lbfgs_tolerance_change = getattr(t_cfg, 'lbfgs_tolerance_change', 1e-12)
        self.lbfgs_history_size = getattr(t_cfg, 'lbfgs_history_size', 50)
        
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        
        self.optimizer_adam = optim.Adam(model.networks.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            model.networks.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=self.lbfgs_maximum_iterations,
            max_eval=self.lbfgs_maximum_evaluations,
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )
        self.last_loss_breakdown = {}
        self.completed_adam_epochs = 0

    @property
    def model(self):
        return self.electro_magnetic_pinn.pinn_architecture_instance

    @property
    def pde_evaluator(self):
        return self.electro_magnetic_pinn.maxwell_pde_loss_instance

    @property
    def interface_evaluator(self):
        return self.electro_magnetic_pinn.interface_loss_instance

    @property
    def visualizer(self):
        return self.electro_magnetic_pinn.visualizer_instance

    @property
    def visualizer_update_interval(self):
        return self.electro_magnetic_pinn.visualization_config.update_interval

    def pretrain_nu_networks(self, domains_data, epochs=None, lr=None):
        if not hasattr(self.model, 'nu_networks') or len(self.model.nu_networks) == 0:
            return

        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)
        epochs = epochs if epochs is not None else getattr(t_cfg, 'pretrain_nu_epochs', 300)
        lr = lr if lr is not None else getattr(t_cfg, 'pretrain_nu_lr', 1e-2)

        print(f"\\n[*] Pre-training Mạng phụ $Net={{\\\\nu}}$ về hằng số ban đầu ({epochs} epochs, lr={lr})...")
        device = next(self.model.parameters()).device
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        
        nu_params = []
        for net in self.model.nu_networks.values():
            nu_params.extend(net.parameters())
            
        opt_nu = optim.Adam(nu_params, lr=lr)
        domains_dict = domains_data.get("domains", domains_data)
        
        for ep in range(epochs):
            opt_nu.zero_grad()
            loss_pre = torch.tensor(0.0, device=device)
            
            for key, data in domains_dict.items():
                if key in self.model.nu_networks:
                    domain = data["domain_object"]
                    pts = data["interior_points"].to(device)
                    
                    if pts.shape[0] > 0:
                        nu_pred_star = self.model.forward_nu_network(key, pts)
                        nu_target_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                        nu_target_star = nu_target_phys / sim_domain.nu0
                        
                        loss_domain = torch.mean((nu_pred_star - nu_target_star) ** 2)
                        loss_pre = loss_pre + loss_domain
                        
            if loss_pre.requires_grad:
                loss_pre.backward()
                opt_nu.step()

    def fit_nu_network_step(self, domains_data, alpha_damping=None, epochs=None, lr=None):
        if not hasattr(self.model, 'nu_networks') or len(self.model.nu_networks) == 0:
            return 0.0

        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)
        alpha_damping = alpha_damping if alpha_damping is not None else getattr(t_cfg, 'fpi_alpha_damping', 0.25)
        epochs = epochs if epochs is not None else getattr(t_cfg, 'fit_nu_epochs', 250)
        lr = lr if lr is not None else getattr(t_cfg, 'fit_nu_lr', 3e-3)

        device = next(self.model.parameters()).device
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        domains_dict = domains_data.get("domains", domains_data)
        
        nu_params = []
        for net in self.model.nu_networks.values():
            nu_params.extend(net.parameters())
        opt_nu = optim.Adam(nu_params, lr=lr)

        max_nu_diff = 0.0
        
        for key, data in domains_dict.items():
            if key in self.model.nu_networks:
                domain = data["domain_object"]
                pts = data["interior_points"].to(device)
                
                if pts.shape[0] > 0:
                    pts_grad = pts.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        Az_star = self.model.forward_network(key, pts_grad)
                        grad_Az = torch.autograd.grad(
                            outputs=Az_star, inputs=pts_grad,
                            grad_outputs=torch.ones_like(Az_star), create_graph=False
                        )[0]
                        Bx_star = grad_Az[:, 1:2] * sim_domain.L0
                        By_star = -grad_Az[:, 0:1] * sim_domain.L0
                        B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * sim_domain.B0
                        
                    with torch.no_grad():
                        nu_BH_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys.detach(), is_nonlinear=True)
                        nu_BH_star = nu_BH_phys / sim_domain.nu0
                        
                        nu_old_star = self.model.forward_nu_network(key, pts)
                        nu_target_star = ((1.0 - alpha_damping) * nu_old_star + alpha_damping * nu_BH_star).detach()
                        
                        diff_val = torch.max(torch.abs(nu_target_star - nu_old_star)).item()
                        max_nu_diff = max(max_nu_diff, diff_val)

                    for _ in range(epochs):
                        opt_nu.zero_grad()
                        nu_pred_star = self.model.forward_nu_network(key, pts)
                        loss_fit = torch.mean((nu_pred_star - nu_target_star) ** 2)
                        loss_fit.backward()
                        opt_nu.step()

        return max_nu_diff

    def compute_loss(self, domains_data):
        device = next(self.model.parameters()).device
        dummy_param = next(self.model.parameters())
        total_loss = (dummy_param * 0.0).sum()

        pde_loss_sum = 0.0
        az_if_sum = 0.0
        bn_if_sum = 0.0
        ht_if_sum = 0.0
        weighted_if_sum = 0.0

        domains_dict = domains_data.get("domains", domains_data)
        shared_dict = domains_data.get("shared_interfaces", {})
        bg_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance.background_domain

        # 1. PDE Loss
        for key, data in domains_dict.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)
                    
                A_z_star = self.model.forward_network(key, pts)
                
                if hasattr(self.model, 'nu_networks') and key in self.model.nu_networks:
                    nu_star_net = self.model.forward_nu_network(key, pts).detach()
                else:
                    nu_star_net = None

                nu_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                hx, hy = domain.evaluate_magnetization_vector(pts)
                hx, hy = hx.to(device), hy.to(device)
                jz = domain.evaluate_current_density(pts).to(device)
                
                res_pde = self.pde_evaluator.compute_linear_pde_residual(
                    xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_star_net=nu_star_net
                )
                fading_w = self.pde_evaluator.compute_fading_weight(domain, pts)
                
                loss_pde_domain = torch.mean((res_pde ** 2) * fading_w)
                w_pde = float(domain.training_config.pde_weight)

                scaled_pde_loss = w_pde * loss_pde_domain
                total_loss = total_loss + scaled_pde_loss
                pde_loss_sum += scaled_pde_loss.item()

            outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
            if outer_pts.shape[0] > 0 and key != 'background':
                if not outer_pts.requires_grad:
                    outer_pts = outer_pts.detach().clone().requires_grad_(True)
                    
                w_if_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
                )
                scaled_if_loss = w_if_loss
                total_loss = total_loss + scaled_if_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_if_loss.item()

        # 2. Shared Interface Loss
        for pair_key, pair_data in shared_dict.items():
            pts = pair_data["interface_points"].to(device)
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)

                w_shared_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                    key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts
                )
                scaled_shared_loss = w_shared_loss
                total_loss = total_loss + scaled_shared_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_shared_loss.item()

        self.last_loss_breakdown = {
            "pde_loss": pde_loss_sum,
            "interface_Az": az_if_sum,
            "interface_Bn": bn_if_sum,
            "interface_Ht": ht_if_sum,
            "weighted_interface": weighted_if_sum,
            "total": total_loss.item()
        }

        return total_loss

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        self.model.train()
        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)

        eta_min = getattr(t_cfg, 'adam_eta_min', 1e-6)
        max_clip_norm = getattr(t_cfg, 'grad_clip_max_norm', 1.0)
        max_non_finite_streak = getattr(t_cfg, 'max_non_finite_streak', 5)
        log_interval = getattr(t_cfg, 'adam_log_interval', 100)

        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_learning_rate_adam
            param_group['lr'] = self.base_learning_rate_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=max(epochs, 1), eta_min=eta_min)
        stop_reason = f"Hoàn thành toàn bộ {epochs} epochs."
        non_finite_streak = 0
        self.completed_adam_epochs = 0
        
        for epoch in range(epochs):
            if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
                domains_data = resample_fn()

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            
            if not torch.isfinite(loss):
                non_finite_streak += 1
                if non_finite_streak >= max_non_finite_streak:
                    stop_reason = f"Dừng sớm do Loss không hữu hạn (NaN/Inf) liên tiếp {max_non_finite_streak} lần."
                    break
                continue
            non_finite_streak = 0
                
            loss.backward()
            if max_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.networks.parameters(), max_norm=max_clip_norm)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            self.completed_adam_epochs = epoch + 1
            current_loss_value = loss.item()
            
            if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0:
                bd = self.last_loss_breakdown
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
                )
                
            if self.visualizer and (epoch + 1) % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
                
        print(f"[*] KẾT THÚC ADAM. Lý do: {stop_reason}")
        return domains_data

    def train_lbfgs(self, max_iterations, domains_data):
        if max_iterations <= 0:
            return domains_data

        self.model.train()
        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)
        log_interval = getattr(t_cfg, 'lbfgs_log_interval', 25)

        self.optimizer_lbfgs = optim.LBFGS(
            self.model.networks.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=max_iterations,
            max_eval=int(max_iterations * 1.25),
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )

        lbfgs_counter = [0]
        
        def closure():
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % log_interval == 0:
                bd = self.last_loss_breakdown
                print(
                    f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {loss.item():.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e}"
                )
            return loss
            
        self.optimizer_lbfgs.step(closure)
        return domains_data
'''

    # =========================================================================
    # 4. CẬP NHẬT: test_simulation.py
    # Minh họa đầy đủ các tham số cấu hình linh hoạt
    # =========================================================================
    files_to_update[test_sim_file] = '''import os
import sys
import shutil
from types import SimpleNamespace

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from electro_magnetic_pinn import ElectroMagneticPINN

def run_nonlinear_simulation_test():
    output_dir = "figure_nonlinear_optimized_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] KHỞI TẠO XPINN PHI TUYẾN - CẤU HÌNH SIÊU THAM SỐ LINH HOẠT...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. CẤU HÌNH KHUNG TOẠ ĐỘ VÀ QUY MÔ VẬT LÝ
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    # 2. CẤU HÌNH LẤY MẪU
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 50
    model.sampler_config.resample_points_before_lbfgs_stage = False

    # 3. TOÀN BỘ CẤU HÌNH THAM SỐ HUẤN LUYỆN
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = True

    # Tổng Epochs & Tốc độ học Adam
    model.training_config.number_of_adam_epochs = 5000
    model.training_config.linear_warmup_epochs_ratio = 0.30       # ~1500 Epochs Warm-up
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-8

    # Cấu hình FPI Outer Loop
    model.training_config.fpi_outer_iterations = 10               # 10 vòng lặp FPI ngoại
    model.training_config.fpi_alpha_damping = 0.25               # Damping factor 0.25
    model.training_config.fpi_tolerance = 1e-5                    # Dung sai FPI

    # Cấu hình Khớp Mạng phụ Net_nu
    model.training_config.pretrain_nu_epochs = 300               # Epochs Pre-train Net_nu ban đầu
    model.training_config.pretrain_nu_lr = 1e-2                  # LR Pre-train Net_nu
    model.training_config.fit_nu_epochs = 250                    # Epochs khớp Net_nu mỗi bước FPI
    model.training_config.fit_nu_lr = 3e-3                       # LR khớp Net_nu mỗi bước FPI

    # Kiểm soát Gradient & Nhật ký
    model.training_config.grad_clip_max_norm = 1.0              # Clip gradient max norm
    model.training_config.adam_eta_min = 1e-6                    # LR tối thiểu của Cosine Annealing
    model.training_config.max_non_finite_streak = 5              # Giới hạn NaN/Inf streak
    model.training_config.adam_log_interval = 100               # Tần suất in log Adam (epochs)
    model.training_config.lbfgs_log_interval = 25               # Tần suất in log L-BFGS (closures)

    # Cấu hình L-BFGS Refinement
    model.training_config.number_of_lbfgs_iterations = 1500
    model.training_config.lbfgs_learning_rate = 0.5
    model.training_config.lbfgs_maximum_evaluations = 2500
    model.training_config.lbfgs_tolerance_gradient = 1e-12
    model.training_config.lbfgs_tolerance_change = 1e-12
    model.training_config.lbfgs_history_size = 50

    # 4. TRỰC QUAN HÓA
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 120
    model.visualization_config.generate_gif_animation = False
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "nonlinear_optimized_training.mp4"

    # 5. CẤU HÌNH CÁC MIỀN VẬT LIỆU
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke", material_name="steel_1008",
        outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    )
    iron_yoke.physics_config.is_nonlinear = True
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_potential_weight = 100.0
    iron_yoke.training_config.interface_bn_weight = 50.0
    iron_yoke.training_config.interface_ht_weight = 50.0
    iron_yoke.training_config.pde_fading_beta = 60.0
    iron_yoke.training_config.number_of_interior_points = 4500
    iron_yoke.training_config.number_of_surface_points = 1500

    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128

    iron_yoke.nu_network_config.hidden_layers = 3
    iron_yoke.nu_network_config.hidden_neurons = 48
    iron_yoke.nu_network_config.activation_function = nn.SiLU()
    iron_yoke.nu_network_config.fourier_features = False

    iron_yoke.build_domain()

    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left", material_name="N30UH",
        outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_potential_weight = 100.0
    magnet_left.training_config.interface_bn_weight = 50.0
    magnet_left.training_config.interface_ht_weight = 50.0
    magnet_left.training_config.number_of_interior_points = 2000
    magnet_left.training_config.number_of_surface_points = 1000

    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128
    magnet_left.build_domain()

    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right", material_name="N30UH",
        outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_potential_weight = 100.0
    magnet_right.training_config.interface_bn_weight = 50.0
    magnet_right.training_config.interface_ht_weight = 50.0
    magnet_right.training_config.number_of_interior_points = 2000
    magnet_right.training_config.number_of_surface_points = 1000

    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128
    magnet_right.build_domain()

    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0, 
        pde_fading_beta=60.0,
        sampling_density_multiplier=1.0, 
        number_of_interior_points=8000,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=64,
        activation_function=nn.SiLU(),
        fourier_features=True, 
        fourier_sigma=1.2, 
        fourier_mapping_size=128
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    model.update_electromagnetic_pinn()
    
    print("[*] Chạy tiến trình huấn luyện PHI TUYẾN...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ phi tuyến...")
    model.plot_fields(resolution=200, save_path=f"{output_dir}/nonlinear_final_fields.png")
    print(f"[✔] HOÀN THÀNH KIỂM THỬ PHI TUYẾN! Kết quả lưu tại: {output_dir}/")

if __name__ == "__main__":
    run_nonlinear_simulation_test()
'''

    # Thực thi ghi đè vào hệ thống
    for file_path, content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật thành công: {os.path.relpath(file_path, project_root)}")

if __name__ == "__main__":
    apply_all_fixes_and_configurations()