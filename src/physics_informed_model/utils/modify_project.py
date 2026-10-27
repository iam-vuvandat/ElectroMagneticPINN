import os

def apply_autograd_sdf_interface_updates():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    physics_dir = os.path.join(project_root, 'physics_domain')
    equations_dir = os.path.join(physics_dir, 'physical_equations')
    training_dir = os.path.join(project_root, 'training_engine')
    root_pinn_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    test_file = os.path.join(project_root, 'tests', 'test_simulation.py')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: physics_domain/physical_equations/interface_loss.py
    # (Dùng Gradient Autograd của SDF để tính trực tiếp nx, ny, tx, ty)
    # =========================================================================
    files_to_update[os.path.join(equations_dir, 'interface_loss.py')] = '''import torch

class InterfaceLoss:
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

    def _compute_normals_and_tangents_from_sdf(self, domain, surf_pts):
        """
        Tính tự động vectơ pháp tuyến (nx, ny) và tiếp tuyến (tx, ty) 
        từ Gradient Autograd của hàm SDF thay vì tra cứu đa giác thô.
        """
        if not surf_pts.requires_grad:
            surf_pts = surf_pts.detach().clone().requires_grad_(True)
            
        sdf_val = domain.compute_signed_distance_field(surf_pts).view(-1, 1)
        
        grad_sdf = torch.autograd.grad(
            outputs=sdf_val,
            inputs=surf_pts,
            grad_outputs=torch.ones_like(sdf_val),
            create_graph=True,
            retain_graph=True
        )[0]
        
        n_norm = torch.sqrt(grad_sdf[:, 0:1]**2 + grad_sdf[:, 1:2]**2 + 1e-12)
        nx = grad_sdf[:, 0:1] / n_norm
        ny = grad_sdf[:, 1:2] / n_norm
        
        tx = -ny
        ty = nx
        
        corner_w = torch.ones_like(nx)
        return nx, ny, tx, ty, corner_w

    def _compute_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            zero = torch.tensor(0.0, device=pts.device)
            return zero, zero, zero, zero

        model = self.electro_magnetic_pinn.pinn_architecture_instance
        nx, ny, tx, ty, corner_w = self._compute_normals_and_tangents_from_sdf(domain_a, pts)

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            _, _, _, _, corner_w_b = self._compute_normals_and_tangents_from_sdf(domain_b, pts)
            corner_w = torch.minimum(corner_w, corner_w_b)

        Az_a = model.forward_network(key_a, pts)
        Az_b = model.forward_network(key_b, pts)

        loss_val = torch.mean((Az_a - Az_b) ** 2)

        grad_a = torch.autograd.grad(
            outputs=Az_a,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_a),
            create_graph=True
        )[0]

        grad_b = torch.autograd.grad(
            outputs=Az_b,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_b),
            create_graph=True
        )[0]

        Bx_a_star = grad_a[:, 1:2] * self.L0
        By_a_star = -grad_a[:, 0:1] * self.L0
        Bx_b_star = grad_b[:, 1:2] * self.L0
        By_b_star = -grad_b[:, 0:1] * self.L0

        Bn_a_star = Bx_a_star * nx + By_a_star * ny
        Bn_b_star = Bx_b_star * nx + By_b_star * ny
        loss_Bn = torch.mean(corner_w * ((Bn_a_star - Bn_b_star) ** 2))

        B_mag_a_phys = torch.sqrt(Bx_a_star**2 + By_a_star**2 + 1e-12) * (self.H0 / self.nu0)
        nu_a_star = domain_a.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_a_phys) / self.nu0
        hx_a, hy_a = domain_a.evaluate_magnetization_vector(pts)
        hx_a_star = hx_a / self.H0
        hy_a_star = hy_a / self.H0

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            B_mag_b_phys = torch.sqrt(Bx_b_star**2 + By_b_star**2 + 1e-12) * (self.H0 / self.nu0)
            nu_b_star = domain_b.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_b_phys) / self.nu0
            hx_b, hy_b = domain_b.evaluate_magnetization_vector(pts)
            hx_b_star = hx_b / self.H0
            hy_b_star = hy_b / self.H0
        else:
            nu_b_star = torch.ones_like(nu_a_star)
            hx_b_star = torch.zeros_like(hx_a_star)
            hy_b_star = torch.zeros_like(hy_a_star)

        Hx_a_star = nu_a_star * Bx_a_star - hx_a_star
        Hy_a_star = nu_a_star * By_a_star - hy_a_star
        Hx_b_star = nu_b_star * Bx_b_star - hx_b_star
        Hy_b_star = nu_b_star * By_b_star - hy_b_star

        Ht_a_star = Hx_a_star * tx + Hy_a_star * ty
        Ht_b_star = Hx_b_star * tx + Hy_b_star * ty
        nu_pair_scale = torch.clamp(torch.maximum(nu_a_star.detach(), nu_b_star.detach()), min=1e-6)
        ht_diff_normalized = (Ht_a_star - Ht_b_star) / nu_pair_scale
        loss_Ht = torch.mean(corner_w * (ht_diff_normalized ** 2))

        w_az = float(getattr(domain_a.training_config, 'interface_potential_weight', 1000.0))
        w_bn = float(getattr(domain_a.training_config, 'interface_bn_weight', 500.0))
        w_ht = float(getattr(domain_a.training_config, 'interface_ht_weight', 500.0))

        weighted_interface = w_az * loss_val + w_bn * loss_Bn + w_ht * loss_Ht
        return weighted_interface, loss_val, loss_Bn, loss_Ht

    def compute_loss_components(self, network_key, domain, surf_pts, background_key='background', contact_tol=1e-4):
        bg_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance.background_domain
        return self._compute_pair_components(network_key, domain, background_key, bg_domain, surf_pts)

    def compute_loss(self, network_key, domain, surf_pts, background_key='background'):
        weighted_interface, _, _, _ = self.compute_loss_components(network_key, domain, surf_pts, background_key)
        return weighted_interface
'''

    # =========================================================================
    # 2. CẬP NHẬT: electro_magnetic_pinn.py (Class Mẹ Hoàn Chỉnh)
    # =========================================================================
    files_to_update[root_pinn_file] = '''import torch
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

            lbfgs_learning_rate=0.8,
            number_of_lbfgs_iterations=800,
            lbfgs_maximum_evaluations=1000,
            lbfgs_tolerance_gradient=1e-8,
            lbfgs_tolerance_change=1e-10,
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
                    outputs=Az_star,
                    inputs=points_tensor,
                    grad_outputs=torch.ones_like(Az_star),
                    create_graph=False
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
            dtype=torch.float32,
            device=device
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
                mu_val = float(getattr(domain.physics_config, 'relative_permeability', 1.0))
                mu_r_tensor[mask] = mu_val

        mu_r_grid = mu_r_tensor.cpu().numpy().reshape(resolution, resolution)

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
        sim_domain = self.simulation_domain_engine_instance
        t_cfg = self.training_config

        total_adam_epochs = t_cfg.number_of_adam_epochs
        warmup_ratio = float(getattr(t_cfg, 'linear_warmup_epochs_ratio', 0.3))
        
        epochs_warmup = int(total_adam_epochs * warmup_ratio) if getattr(t_cfg, 'enable_linear_warmup_stage', True) else 0
        epochs_adaptation = (total_adam_epochs - epochs_warmup) if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) else 0

        if getattr(t_cfg, 'enable_linear_warmup_stage', True) and epochs_warmup > 0:
            print(f"\\n--- [STAGE 1A] Adam Linear Warm-up ({epochs_warmup} Epochs) ---")
            for domain in sim_domain.domains_list:
                domain.physics_config.is_nonlinear = False

            domains_data = self.training_manager_instance.train_adam(
                epochs=epochs_warmup, 
                domains_data=domains_data,
                dynamic_resampling=self.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn,
                resample_interval=self.sampler_config.resample_interval
            )
        else:
            print("\\n--- [STAGE 1A] Adam Linear Warm-up: Bỏ qua ---")

        if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) and epochs_adaptation > 0:
            print(f"\\n--- [STAGE 1B] Adam Non-linear Adaptation ({epochs_adaptation} Epochs) ---")
            for domain in sim_domain.domains_list:
                if getattr(domain.physics_config, 'bh_curve_data', None) is not None:
                    domain.physics_config.is_nonlinear = True

            domains_data = self.training_manager_instance.train_adam(
                epochs=epochs_adaptation, 
                domains_data=domains_data,
                dynamic_resampling=self.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn,
                resample_interval=self.sampler_config.resample_interval
            )
        else:
            print("\\n--- [STAGE 1B] Adam Non-linear Adaptation: Bỏ qua ---")

        if getattr(t_cfg, 'enable_lbfgs_refinement_stage', True) and t_cfg.number_of_lbfgs_iterations > 0:
            if getattr(self.sampler_config, 'resample_points_before_lbfgs_stage', True):
                domains_data = resample_data_fn()

            lbfgs_iters = int(t_cfg.number_of_lbfgs_iterations)
            print(f"\\n--- [STAGE 2] L-BFGS Refinement (Tối đa {lbfgs_iters} Iterations) ---")
            self.training_manager_instance.train_lbfgs(
                max_iterations=lbfgs_iters, 
                domains_data=domains_data
            )
        else:
            print("\\n--- [STAGE 2] L-BFGS Refinement: Bỏ qua ---")
            
        if self.visualizer_instance:
            self.visualizer_instance.generate_animations()
'''

    # Thực hiện ghi nội dung cập nhật
    for file_path, code_content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        print(f"[*] Đã ghi và cập nhật thành công: {os.path.relpath(file_path, project_root)}")

    print("\n[✔] HOÀN THÀNH CHUYỂN ĐỔI SANG GRADIENT AUTOGRAD SDF CHO INTERFACE LOSS!")

if __name__ == "__main__":
    apply_autograd_sdf_interface_updates()