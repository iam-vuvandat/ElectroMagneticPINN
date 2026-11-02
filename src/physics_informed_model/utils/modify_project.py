import os

def apply_visualization_config_modifications():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    pinn_core_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    visualizer_file = os.path.join(project_root, 'utils', 'training_visualizer.py')

    # =========================================================================
    # 1. CẬP NHẬT: electro_magnetic_pinn.py
    # =========================================================================
    pinn_core_code = '''import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from types import SimpleNamespace
from architectures.xpinn_architecture import XPINNArchitecture
from training_engine.training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from physics_domain.physical_equations.interface_loss import InterfaceLoss
from simulation_domain_engine.simulation_domain import SimulationDomain
from physics_domain.collocation_sampler import CollocationSampler
from utils.training_visualizer import TrainingVisualizer
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

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

            fpi_outer_iterations=10,
            fpi_alpha_damping=1.0,
            fpi_tolerance=1e-5,

            pretrain_nu_epochs=300,
            pretrain_nu_lr=1e-2,
            fit_nu_epochs=250,
            fit_nu_lr=3e-3,

            grad_clip_max_norm=1.0,
            adam_eta_min=1e-6,
            max_non_finite_streak=5,
            adam_log_interval=100,
            lbfgs_log_interval=25,

            lbfgs_learning_rate=0.5,
            number_of_lbfgs_iterations=1500,
            lbfgs_maximum_evaluations=2500,
            lbfgs_tolerance_gradient=1e-12,
            lbfgs_tolerance_change=1e-12,
            lbfgs_history_size=50
        )
        
        # Cấu hình trực quan hóa mở rộng cho toàn bộ 8 đồ thị
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
            mp4_quality=8,

            contour_levels=60,

            # 1. Sai số PDE Residual Error (axs[0, 1])
            res_scale="log",          
            res_vmin=1e-4,            
            res_vmax=1e1,             
            res_cmap="Reds",

            # 2. Độ từ thẩm tương đối mu_r (axs[0, 2]) - TUYẾN TÍNH CỐ ĐỊNH 0 - 3000
            mur_scale="linear",       # Chuyển sang tuyến tính
            mur_vmin=0.0,             # Min = 0
            mur_vmax=3000.0,          # Max = 3000 cố định
            mur_cmap="turbo",         # Colormap đa sắc dải rộng

            # 3. Thế vectơ từ A_z (axs[1, 0])
            az_scale="linear",        
            az_vmin=None,             
            az_vmax=None,             
            az_cmap="jet",

            # 4. Mật độ từ thông |B| (axs[1, 1] & axs[1, 2]) - CỐ ĐỊNH SCALE MIN-MAX
            b_mag_scale="linear",     
            b_mag_vmin=0.0,           # Min cố định = 0.0 T
            b_mag_vmax=1.8,           # Max cố định = 1.8 T
            b_mag_cmap="jet",

            # 5. Thành phần từ trường B_y (cho plot_fields)
            by_scale="linear",        
            by_vmin=None,             
            by_vmax=None,             
            by_cmap="coolwarm"
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
        vis_cfg = self.visualization_config

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
        B_mag_pts = torch.tensor(B_mag_grid.ravel(), dtype=torch.float32, device=device).unsqueeze(1)

        for domain in sim_domain.domains_list:
            if getattr(domain, 'domain_name', '') != 'background':
                mask = (mat_ids == float(domain.domain_id))
                if mask.any():
                    if getattr(domain.physics_config, 'is_nonlinear', False):
                        nu_phys = domain.evaluate_reluctivity(eval_pts[mask], B_magnitude_tensor=B_mag_pts[mask], is_nonlinear=True)
                        mu_r_tensor[mask] = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys.squeeze(), min=1e-6))
                    else:
                        mu_val = float(getattr(domain.physics_config, 'relative_permeability', 1.0))
                        mu_r_tensor[mask] = mu_val

        mu_r_grid = mu_r_tensor.detach().cpu().numpy().reshape(resolution, resolution)

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        n_levels = getattr(vis_cfg, 'contour_levels', 60)
        
        # --- 1. A_z Plot ---
        az_vmin = vis_cfg.az_vmin if vis_cfg.az_vmin is not None else float(np.min(A_z_grid))
        az_vmax = vis_cfg.az_vmax if vis_cfg.az_vmax is not None else float(np.max(A_z_grid))
        levels_az = np.linspace(az_vmin, az_vmax, n_levels) if az_vmin != az_vmax else n_levels
        c0 = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=levels_az, cmap=vis_cfg.az_cmap)
        fig.colorbar(c0, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
        axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
        axs[0, 0].set_aspect('equal')

        # --- 2. |B| Magnitude Plot (Cố định Scale) ---
        b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
        b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else 1.8
        b_mag_clamped = np.clip(B_mag_grid, b_min, b_max)
        levels_b = np.linspace(b_min, b_max, n_levels)
        c1 = axs[0, 1].contourf(X_grid, Y_grid, b_mag_clamped, levels=levels_b, cmap=vis_cfg.b_mag_cmap, vmin=b_min, vmax=b_max)
        fig.colorbar(c1, ax=axs[0, 1], label=r"$|B|$ (T)")
        axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
        axs[0, 1].set_aspect('equal')

        # --- 3. Relative Permeability \mu_r Plot (Linear 0 - 3000) ---
        v_min_mur = vis_cfg.mur_vmin if vis_cfg.mur_vmin is not None else 0.0
        v_max_mur = vis_cfg.mur_vmax if vis_cfg.mur_vmax is not None else 3000.0
        mu_r_clamped = np.clip(mu_r_grid, v_min_mur, v_max_mur)
        
        if getattr(vis_cfg, 'mur_scale', 'linear') == 'log':
            levels_mur = np.logspace(np.log10(max(v_min_mur, 1e-3)), np.log10(v_max_mur), n_levels)
            norm_mur = LogNorm(vmin=max(v_min_mur, 1.0), vmax=v_max_mur)
            lbl_mur = r"$\mu_r$ (Log Scale)"
        else:
            levels_mur = np.linspace(v_min_mur, v_max_mur, n_levels)
            norm_mur = Normalize(vmin=v_min_mur, vmax=v_max_mur)
            lbl_mur = r"$\mu_r$ (Linear Scale: 0 - 3000)"

        c2 = axs[1, 0].contourf(X_grid, Y_grid, mu_r_clamped, levels=levels_mur, cmap=vis_cfg.mur_cmap, norm=norm_mur)
        fig.colorbar(c2, ax=axs[1, 0], label=lbl_mur)
        axs[1, 0].set_title(r"Relative Permeability Distribution ($\mu_r$)")
        axs[1, 0].set_aspect('equal')

        # --- 4. B_y Component Plot ---
        by_vmin = vis_cfg.by_vmin if vis_cfg.by_vmin is not None else float(np.min(B_y_grid))
        by_vmax = vis_cfg.by_vmax if vis_cfg.by_vmax is not None else float(np.max(B_y_grid))
        levels_by = np.linspace(by_vmin, by_vmax, n_levels) if by_vmin != by_vmax else n_levels
        c3 = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=levels_by, cmap=vis_cfg.by_cmap)
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

        pre_ep = getattr(t_cfg, 'pretrain_nu_epochs', 300)
        pre_lr = getattr(t_cfg, 'pretrain_nu_lr', 1e-2)
        self.training_manager_instance.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)

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
            alpha_damp = float(getattr(t_cfg, 'fpi_alpha_damping', 1.0))
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
    # 2. CẬP NHẬT: utils/training_visualizer.py
    # =========================================================================
    visualizer_code = '''import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import imageio
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

class TrainingVisualizer:
    def __init__(self, electro_magnetic_pinn=None, parent_pinn=None):
        self.electro_magnetic_pinn = electro_magnetic_pinn if electro_magnetic_pinn is not None else parent_pinn
        self.parent_pinn = self.electro_magnetic_pinn
        
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        vis_cfg = self.electro_magnetic_pinn.visualization_config
        
        self.output_directory = vis_cfg.output_directory
        self.x_bounds = sim_domain.x_boundaries_tuple
        self.y_bounds = sim_domain.y_boundaries_tuple
        self.resolution = vis_cfg.resolution
        
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
        
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        pde_evaluator = self.electro_magnetic_pinn.maxwell_pde_loss_instance
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        vis_cfg = self.electro_magnetic_pinn.visualization_config

        model.eval()
        computation_device = next(model.parameters()).device
        eval_points = self.eval_points_tensor.to(computation_device).detach().clone().requires_grad_(True)
        
        phys_props = sim_domain.evaluate_global_physical_properties(eval_points)
        mat_id = phys_props["material_classification"]
        
        A_z_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_x_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_y_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        nu_phys_all = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        residual_total = torch.zeros((eval_points.shape[0], 1), device=computation_device)

        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{d_id}'
            mask_bool = (mat_id == float(d_id))
            if mask_bool.any():
                Az_sub_star = model.forward_network(net_key, eval_points)
                grad_sub_star = torch.autograd.grad(
                    outputs=Az_sub_star, inputs=eval_points,
                    grad_outputs=torch.ones_like(Az_sub_star), create_graph=True
                )[0]
                Az_sub_phys = Az_sub_star * sim_domain.A0
                Bx_sub_phys = grad_sub_star[:, 1:2] * sim_domain.A0
                By_sub_phys = -grad_sub_star[:, 0:1] * sim_domain.A0

                if getattr(domain.physics_config, 'is_nonlinear', False):
                    B_mag_sub = torch.sqrt(Bx_sub_phys**2 + By_sub_phys**2 + 1e-12)
                    nu_dom = domain.evaluate_reluctivity(eval_points, B_magnitude_tensor=B_mag_sub, is_nonlinear=True).to(computation_device)
                else:
                    nu_dom = domain.evaluate_reluctivity(eval_points, is_nonlinear=False).to(computation_device)

                hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
                hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
                jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

                res_pde = pde_evaluator.compute_linear_pde_residual(
                    eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom
                )

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
                nu_phys_all = torch.where(mask_bool, nu_dom.detach(), nu_phys_all)
                residual_total = torch.where(mask_bool, torch.abs(res_pde.detach()), residual_total)
        
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
        mu_r_numpy = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys_all, min=1e-6)).cpu().numpy().reshape(self.resolution, self.resolution)
        residual_numpy = residual_total.cpu().numpy().reshape(self.resolution, self.resolution)
        
        fig, axs = plt.subplots(2, 4, figsize=(23, 10))
        n_levels = getattr(vis_cfg, 'contour_levels', 60)

        # --- SUBPLOT 1: axs[0, 0] - Categorized Collocation Points ---
        ax_pts = axs[0, 0]
        ax_pts.set_title("Categorized Collocation Points")
        domain_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
        pair_colors = ['#d62728', '#e377c2', '#bcbd22', '#17becf', '#ff1493']

        domains_dict = domains_data.get("domains", domains_data)
        shared_dict = domains_data.get("shared_interfaces", {})

        for idx, (key, d) in enumerate(domains_dict.items()):
            dom_obj = d["domain_object"]
            d_name = getattr(dom_obj, "domain_name", key)
            c_dom = domain_colors[idx % len(domain_colors)]
            
            int_pts = d.get("interior_points")
            if int_pts is not None and int_pts.shape[0] > 0:
                pts_np = int_pts.detach().cpu().numpy()
                ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=2.5, color=c_dom, alpha=0.35, label=f"Int: {d_name}")

        for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
            if_pts = pair_data.get("interface_points")
            if if_pts is not None and if_pts.shape[0] > 0:
                pts_np = if_pts.detach().cpu().numpy()
                c_pair = pair_colors[p_idx % len(pair_colors)]
                name_a = getattr(pair_data["domain_a"], "domain_name", pair_data["key_a"])
                name_b = getattr(pair_data["domain_b"], "domain_name", pair_data["key_b"])
                ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=20, color=c_pair, marker='x', linewidths=1.5, label=f"Interface: {name_a} ↔ {name_b}")

        ax_pts.legend(loc='upper right', fontsize=6.5, framealpha=0.85)
        ax_pts.set_xlim(self.x_bounds)
        ax_pts.set_ylim(self.y_bounds)
        ax_pts.set_aspect('equal')
        
        # --- SUBPLOT 2: axs[0, 1] - PDE Residual Error ---
        res_min = vis_cfg.res_vmin if vis_cfg.res_vmin is not None else float(np.min(residual_numpy))
        res_max = vis_cfg.res_vmax if vis_cfg.res_vmax is not None else float(np.max(residual_numpy))

        if getattr(vis_cfg, 'res_scale', 'log') == 'log':
            res_min_clean = max(res_min, 1e-6)
            res_max_clean = max(res_max, res_min_clean * 10.0)
            res_abs = np.clip(np.abs(residual_numpy), res_min_clean, res_max_clean)
            norm_res = LogNorm(vmin=res_min_clean, vmax=res_max_clean)
            levels_res = np.logspace(np.log10(res_min_clean), np.log10(res_max_clean), n_levels)
        else:
            res_abs = residual_numpy
            norm_res = Normalize(vmin=res_min, vmax=res_max)
            levels_res = np.linspace(res_min, res_max, n_levels)

        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, res_abs, levels=levels_res, cmap=vis_cfg.res_cmap, norm=norm_res)
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')

        # --- SUBPLOT 3: axs[0, 2] - Permeability Distribution (\mu_r) - LINEAR SCALE (0 - 3000) ---
        mur_min = vis_cfg.mur_vmin if vis_cfg.mur_vmin is not None else 0.0
        mur_max = vis_cfg.mur_vmax if vis_cfg.mur_vmax is not None else 3000.0
        mu_r_clamped = np.clip(mu_r_numpy, mur_min, mur_max)

        if getattr(vis_cfg, 'mur_scale', 'linear') == 'log':
            mur_min_clean = max(mur_min, 1e-3)
            mur_max_clean = max(mur_max, mur_min_clean * 2.0)
            norm_mur = LogNorm(vmin=mur_min_clean, vmax=mur_max_clean)
            levels_mur = np.logspace(np.log10(mur_min_clean), np.log10(mur_max_clean), n_levels)
            lbl_mur = r"$\mu_r$ (Log Scale)"
        else:
            norm_mur = Normalize(vmin=mur_min, vmax=mur_max)
            levels_mur = np.linspace(mur_min, mur_max, n_levels)
            lbl_mur = r"$\mu_r$ (Linear Scale: 0 - 3000)"

        contour_mur = axs[0, 2].contourf(
            self.X_grid, self.Y_grid, mu_r_clamped, levels=levels_mur,
            cmap=vis_cfg.mur_cmap, norm=norm_mur
        )
        fig.colorbar(contour_mur, ax=axs[0, 2], label=lbl_mur)
        axs[0, 2].set_title(r"Permeability Distribution ($\mu_r$)")
        axs[0, 2].set_aspect('equal')
        
        # --- SUBPLOT 4: axs[0, 3] - Loss Optimization History ---
        axs[0, 3].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 3].set_yscale('log')
        axs[0, 3].set_title("Loss Optimization")
        axs[0, 3].set_xlabel("Step / Epoch")
        axs[0, 3].set_ylabel("Loss")
        axs[0, 3].grid(True, which="both", ls="-", alpha=0.2)
        
        # --- SUBPLOT 5: axs[1, 0] - Magnetic Vector Potential A_z ---
        az_min = vis_cfg.az_vmin if vis_cfg.az_vmin is not None else float(np.min(A_z_numpy))
        az_max = vis_cfg.az_vmax if vis_cfg.az_vmax is not None else float(np.max(A_z_numpy))
        levels_az = np.linspace(az_min, az_max, n_levels) if az_min != az_max else n_levels

        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=levels_az, cmap=vis_cfg.az_cmap)
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        # --- SUBPLOT 6: axs[1, 1] - Flux Density Magnitude |B| - CỐ ĐỊNH SCALE MIN-MAX ---
        b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
        b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else 1.8
        b_mag_clamped = np.clip(B_mag_numpy, b_min, b_max)
        levels_b = np.linspace(b_min, b_max, n_levels)

        contour_b = axs[1, 1].contourf(
            self.X_grid, self.Y_grid, b_mag_clamped, levels=levels_b,
            cmap=vis_cfg.b_mag_cmap, vmin=b_min, vmax=b_max
        )
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title(f"Flux Density Magnitude ($|B|$ fixed [{b_min}-{b_max}T])")
        axs[1, 1].set_aspect('equal')

        # --- SUBPLOT 7: axs[1, 2] - Flux Density Vectors (B) ---
        contour_b_bg = axs[1, 2].contourf(
            self.X_grid, self.Y_grid, b_mag_clamped, levels=levels_b,
            cmap=vis_cfg.b_mag_cmap, alpha=0.3, vmin=b_min, vmax=b_max
        )
        fig.colorbar(contour_b_bg, ax=axs[1, 2], label="|B| (T)")
        step = max(1, self.resolution // 20)
        B_sub = np.maximum(B_mag_numpy[::step, ::step], 1e-12)
        axs[1, 2].quiver(
            self.X_grid[::step, ::step], self.Y_grid[::step, ::step], 
            B_x_numpy[::step, ::step] / B_sub, B_y_numpy[::step, ::step] / B_sub, 
            scale=30.0, color='black', pivot='mid'
        )
        axs[1, 2].set_title("Flux Density Vectors (B)")
        axs[1, 2].set_aspect('equal')

        # --- SUBPLOT 8: axs[1, 3] - Material Classification ---
        mat_id_numpy = mat_id.cpu().numpy().reshape(self.resolution, self.resolution)
        im_mat = axs[1, 3].imshow(mat_id_numpy, extent=[self.x_bounds[0], self.x_bounds[1], self.y_bounds[0], self.y_bounds[1]], origin='lower', interpolation='nearest', cmap='tab10')
        fig.colorbar(im_mat, ax=axs[1, 3], label="Domain ID")
        axs[1, 3].set_title("Material Classification")
        axs[1, 3].set_aspect('equal')
        
        fig.suptitle(f"XPINN Training Process - Step {epoch}", fontsize=16)
        plt.tight_layout()
        
        frame_name = os.path.join(self.output_directory, f"frame_{len(self.frame_paths) + 1:05d}.png")
        plt.savefig(frame_name, dpi=100)
        plt.close(fig)
        
        self.frame_paths.append(frame_name)
        model.train()

    def generate_gif(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'gif_filename', 'training_process.gif')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        if not self.frame_paths:
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = [imageio.imread(f) for f in self.frame_paths if os.path.exists(f)]
        imageio.mimsave(full_gif_path, images, fps=fps, loop=0)

    def generate_mp4(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        codec = getattr(cfg, 'mp4_codec', 'libx264')
        quality = getattr(cfg, 'mp4_quality', 8)

        valid_frames = [f for f in self.frame_paths if os.path.exists(f)]
        if not valid_frames:
            return

        full_mp4_path = os.path.join(self.output_directory, output_filename)
        try:
            writer = imageio.get_writer(full_mp4_path, fps=fps, codec=codec, pixelformat='yuv420p', quality=quality)
            for f in valid_frames:
                writer.append_data(imageio.imread(f))
            writer.close()
        except Exception as exc:
            pass

    def generate_animations(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        if getattr(cfg, 'generate_gif_animation', True):
            self.generate_gif()
        if getattr(cfg, 'generate_mp4_video', True):
            self.generate_mp4()
'''

    files_to_update = {
        pinn_core_file: pinn_core_code,
        visualizer_file: visualizer_code
    }

    for file_path, content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật thành công: {os.path.relpath(file_path, project_root)}")


if __name__ == "__main__":
    apply_visualization_config_modifications()