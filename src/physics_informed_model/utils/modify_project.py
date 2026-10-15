import os
import sys

def modify_project():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))
    
    pinn_path = os.path.join(project_root, 'electro_magnetic_pinn.py')
    visualizer_path = os.path.join(project_root, 'utils', 'training_visualizer.py')
    test_sim_path = os.path.join(project_root, 'tests', 'test_simulation.py')

    pinn_content = '''import torch
import torch.nn as nn
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
            resample_before_lbfgs=True
        )
        
        self.training_config = SimpleNamespace(
            learning_rate_adam=1e-3,
            target_loss=1e-5,
            epochs_adam=2500,
            epochs_lbfgs=800,
            lbfgs_learning_rate=0.8,
            lbfgs_maximum_iterations=800,
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

    def execute_training_process(self):
        if self.pinn_architecture_instance is None:
            self._build_system()

        def resample_data_fn():
            return self.collocation_sampler_instance.generate_all_domain_points()

        domains_data = resample_data_fn()
        for key, d in domains_data.items():
            print(
                f"  - Domain [{key} ({d['domain_object'].domain_name})]: "
                f"interior = {d['interior_points'].shape[0]}, surface = {d['surface_points'].shape[0]}"
            )
        
        print(f"--- Standard Adam Training ({self.training_config.epochs_adam} Epochs) ---")
        domains_data = self.training_manager_instance.train_adam(
            epochs=self.training_config.epochs_adam, 
            domains_data=domains_data,
            dynamic_resampling=self.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn,
            resample_interval=self.sampler_config.resample_interval
        )
        
        if getattr(self.sampler_config, 'resample_before_lbfgs', True):
            domains_data = resample_data_fn()

        lbfgs_iters = int(self.training_config.epochs_lbfgs)
        print(f"--- L-BFGS Refinement (Max {lbfgs_iters} Iterations) ---")
        self.training_manager_instance.train_lbfgs(
            max_iterations=lbfgs_iters, 
            domains_data=domains_data
        )
        
        if self.visualizer_instance:
            self.visualizer_instance.generate_animations()

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor = points_tensor.to(self.computation_device).detach().clone().requires_grad_(True)
        
        props = self.simulation_domain_engine_instance.evaluate_global_physical_properties(points_tensor)
        mat_id = props["material_classification"]
        
        A_z_phys = torch.zeros((points_tensor.shape[0], 1), device=self.computation_device)
        B_x_phys = torch.zeros((points_tensor.shape[0], 1), device=self.computation_device)
        B_y_phys = torch.zeros((points_tensor.shape[0], 1), device=self.computation_device)
        
        for domain in self.simulation_domain_engine_instance.domains_list:
            d_id = domain.domain_id
            if getattr(domain, 'domain_name', '') == 'background':
                net_key = 'background'
            else:
                net_key = f'domain_{d_id}'
                
            mask_bool = (mat_id == float(d_id))
            if mask_bool.any():
                Az_sub_star = self.pinn_architecture_instance.forward_network(net_key, points_tensor)
                grad_sub_star = torch.autograd.grad(
                    outputs=Az_sub_star,
                    inputs=points_tensor,
                    grad_outputs=torch.ones_like(Az_sub_star),
                    create_graph=False
                )[0]
                Az_sub_phys = Az_sub_star * self.A0
                Bx_sub_phys = grad_sub_star[:, 1:2] * self.A0
                By_sub_phys = -grad_sub_star[:, 0:1] * self.A0

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
        
        return A_z_phys.detach(), B_x_phys.detach(), B_y_phys.detach()
'''

    visualizer_content = '''import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio

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
        
        model.eval()
        computation_device = next(model.parameters()).device
        eval_points = self.eval_points_tensor.to(computation_device).detach().clone().requires_grad_(True)
        
        phys_props = sim_domain.evaluate_global_physical_properties(eval_points)
        mat_id = phys_props["material_classification"]
        
        A_z_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_x_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_y_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        residual_total = torch.zeros((eval_points.shape[0], 1), device=computation_device)

        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{d_id}'
            mask_bool = (mat_id == float(d_id))
            if mask_bool.any():
                Az_sub_star = model.forward_network(net_key, eval_points)
                grad_sub_star = torch.autograd.grad(
                    outputs=Az_sub_star,
                    inputs=eval_points,
                    grad_outputs=torch.ones_like(Az_sub_star),
                    create_graph=True
                )[0]
                Az_sub_phys = Az_sub_star * sim_domain.A0
                Bx_sub_phys = grad_sub_star[:, 1:2] * sim_domain.A0
                By_sub_phys = -grad_sub_star[:, 0:1] * sim_domain.A0

                nu_dom = domain.evaluate_reluctivity(eval_points).to(computation_device)
                hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
                hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
                jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

                res_sub = pde_evaluator.compute_residual(
                    eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom
                )

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
                residual_total = torch.where(mask_bool, torch.abs(res_sub.detach()), residual_total)
        
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
        residual_numpy = residual_total.cpu().numpy().reshape(self.resolution, self.resolution)
        
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
        
        res_abs = np.clip(np.abs(residual_numpy), 1e-6, 1e2)
        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, res_abs, levels=50, cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=1e1))
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')
        
        axs[0, 2].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 2].set_yscale('log')
        axs[0, 2].set_title("Loss Optimization")
        axs[0, 2].set_xlabel("Step / Epoch")
        axs[0, 2].set_ylabel("Loss")
        axs[0, 2].grid(True, which="both", ls="-", alpha=0.2)
        
        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=50, cmap="jet")
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        b_max_val = max(float(np.max(B_mag_numpy)), 1e-3)
        levels_b = np.linspace(0.0, b_max_val, 50)
        contour_b = axs[1, 1].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=levels_b, cmap="jet", vmin=0.0, vmax=b_max_val)
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title("Flux Density Magnitude ($|B|$)")
        axs[1, 1].set_aspect('equal')

        contour_b_bg = axs[1, 2].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=levels_b, cmap="jet", alpha=0.3, vmin=0.0, vmax=b_max_val)
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
        
        fig.suptitle(f"PINN Training Process - Step {epoch}", fontsize=16)
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
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps, loop=0)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")

    def generate_mp4(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        codec = getattr(cfg, 'mp4_codec', 'libx264')
        quality = getattr(cfg, 'mp4_quality', 8)

        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo MP4.")
            return

        valid_frames = [f for f in self.frame_paths if os.path.exists(f)]
        if not valid_frames:
            print("[!] CẢNH BÁO: Tất cả frame paths đều không tồn tại trên đĩa.")
            return

        full_mp4_path = os.path.join(self.output_directory, output_filename)
        print(f"[*] Đang xuất Video MP4 ({len(valid_frames)} frames, {fps} fps)...")
        
        exported = False
        try:
            writer = imageio.get_writer(
                full_mp4_path,
                fps=fps,
                codec=codec,
                pixelformat='yuv420p',
                quality=quality
            )
            for f in valid_frames:
                writer.append_data(imageio.imread(f))
            writer.close()
            exported = True
            print(f"[*] Đã xuất MP4 Video thành công (imageio) tại: {full_mp4_path}")
        except Exception as exc:
            print(f"[!] Không thể xuất MP4 bằng imageio-ffmpeg ({exc}). Chuyển sang OpenCV fallback...")

        if not exported:
            try:
                import cv2
                sample = cv2.imread(valid_frames[0])
                height, width, _ = sample.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(full_mp4_path, fourcc, float(fps), (width, height))
                for f in valid_frames:
                    img = cv2.imread(f)
                    video_writer.write(img)
                video_writer.release()
                print(f"[*] Đã xuất MP4 Video thành công (OpenCV) tại: {full_mp4_path}")
            except Exception as cv_exc:
                print(f"[!] Lỗi khi xuất video MP4 bằng OpenCV: {cv_exc}")

    def generate_animations(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        gen_gif = getattr(cfg, 'generate_gif_animation', True)
        gen_mp4 = getattr(cfg, 'generate_mp4_video', True)

        if gen_gif:
            self.generate_gif()
        if gen_mp4:
            self.generate_mp4()
'''

    test_sim_content = '''import os
import sys
import shutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation_domain_engine.material_domain.material_domain import MaterialDomain
from electro_magnetic_pinn import ElectroMagneticPINN

def main():
    if os.path.exists('figure'):
        shutil.rmtree('figure')
    os.makedirs('figure')
    print("version 10.5 - Dual Magnets XPINN Simulation without Seam Border")
    
    model = ElectroMagneticPINN()
    
    max_coercive_field = 800000.0
    model.simulation_domain_engine_instance.physics_config.scale_H0 = max_coercive_field * 1.2
    model.simulation_domain_engine_instance.physics_config.scale_L0 = 0.1 * 1.2
    
    model.simulation_domain_engine_instance.x_boundaries_tuple = (-0.1, 0.1)
    model.simulation_domain_engine_instance.y_boundaries_tuple = (-0.1, 0.1)
    
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_before_lbfgs = True
    
    model.training_config.epochs_adam = 3500
    model.training_config.epochs_lbfgs = 1500
    model.training_config.learning_rate_adam = 1e-3
    model.training_config.target_loss = 1e-7
    
    model.training_config.lbfgs_learning_rate = 0.8
    model.training_config.lbfgs_maximum_iterations = 1500
    model.training_config.lbfgs_maximum_evaluations = 1875
    model.training_config.lbfgs_tolerance_gradient = 1e-8
    model.training_config.lbfgs_tolerance_change = 1e-10
    model.training_config.lbfgs_history_size = 50
    
    model.visualization_config.active = True
    model.visualization_config.update_interval = 50 
    model.visualization_config.output_directory = "figure"
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "training_process.mp4"
    model.visualization_config.animation_fps = 5
    model.visualization_config.gif_fps = 5
    
    i_magnet_left = MaterialDomain(domain_name="i_bar_left")
    i_magnet_left.geometry_config.outline = [
        [-0.04, -0.04], 
        [-0.02, -0.04], 
        [-0.02, 0.04], 
        [-0.04, 0.04]
    ]
    i_magnet_left.physics_config.relative_permeability = 1.05
    i_magnet_left.physics_config.coercive = [0.0, 800000.0]
    i_magnet_left.training_config.pde_weight = 1.0
    i_magnet_left.training_config.interface_weight = 1.0
    i_magnet_left.training_config.interface_potential_weight = 1000.0
    i_magnet_left.training_config.interface_bn_weight = 350.0
    i_magnet_left.training_config.interface_ht_weight = 350.0
    i_magnet_left.training_config.pde_fading_beta = 90.0
    i_magnet_left.training_config.number_of_interior_points = 1200
    i_magnet_left.training_config.number_of_surface_points = 800
    i_magnet_left.network_config.hidden_layers = 4
    i_magnet_left.network_config.hidden_neurons = 64
    i_magnet_left.network_config.activation_function = nn.SiLU()
    i_magnet_left.network_config.fourier_features = True
    i_magnet_left.network_config.fourier_sigma = 1.2
    i_magnet_left.network_config.fourier_mapping_size = 128
    i_magnet_left.build_domain()

    i_magnet_right = MaterialDomain(domain_name="i_bar_right")
    i_magnet_right.geometry_config.outline = [
        [0.02, -0.04], 
        [0.04, -0.04], 
        [0.04, 0.04], 
        [0.02, 0.04]
    ]
    i_magnet_right.physics_config.relative_permeability = 1.05
    i_magnet_right.physics_config.coercive = [0.0, -800000.0]
    i_magnet_right.training_config.pde_weight = 1.0
    i_magnet_right.training_config.interface_weight = 1.0
    i_magnet_right.training_config.interface_potential_weight = 1000.0
    i_magnet_right.training_config.interface_bn_weight = 350.0
    i_magnet_right.training_config.interface_ht_weight = 350.0
    i_magnet_right.training_config.pde_fading_beta = 90.0
    i_magnet_right.training_config.number_of_interior_points = 1200
    i_magnet_right.training_config.number_of_surface_points = 800
    i_magnet_right.network_config.hidden_layers = 4
    i_magnet_right.network_config.hidden_neurons = 64
    i_magnet_right.network_config.activation_function = nn.SiLU()
    i_magnet_right.network_config.fourier_features = True
    i_magnet_right.network_config.fourier_sigma = 1.2
    i_magnet_right.network_config.fourier_mapping_size = 128
    i_magnet_right.build_domain()

    model.simulation_domain_engine_instance.add_material_domain(i_magnet_left)
    model.simulation_domain_engine_instance.add_material_domain(i_magnet_right)

    bg_training_config = type('', (), {})()
    bg_training_config.pde_weight = 1.0
    bg_training_config.interface_weight = 1.0
    bg_training_config.pde_fading_beta = 90.0
    bg_training_config.sampling_density_multiplier = 1.0
    bg_training_config.number_of_interior_points = 5000
    bg_training_config.number_of_surface_points = 0

    bg_network_config = type('', (), {})()
    bg_network_config.hidden_layers = 4
    bg_network_config.hidden_neurons = 64
    bg_network_config.activation_function = nn.SiLU()
    bg_network_config.fourier_features = True
    bg_network_config.fourier_sigma = 1.2
    bg_network_config.fourier_mapping_size = 128

    model.simulation_domain_engine_instance.build_background_domain(
        domain_name="background",
        training_config=bg_training_config,
        network_config=bg_network_config
    )

    model.update_electromagnetic_pinn()

    xb = model.simulation_domain_engine_instance.x_boundaries_tuple
    yb = model.simulation_domain_engine_instance.y_boundaries_tuple
    
    model.simulation_domain_engine_instance.plot_problem_definition(resolution=100)

    model.execute_training_process()
    
    resolution = 180
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(xy_points_tensor)
    assert torch.all(torch.isfinite(A_z_pred)) and torch.all(torch.isfinite(B_x_pred)) and torch.all(torch.isfinite(B_y_pred)), "Kết quả trường từ không hữu hạn!"
    
    A_z_grid = A_z_pred.cpu().numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.cpu().numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.cpu().numpy().reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)
    
    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    contour_az = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=60, cmap="jet")
    fig1.colorbar(contour_az, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
    axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
    axs[0, 0].set_aspect('equal')
    
    b_max_val = max(float(np.max(B_mag_grid)), 1e-3)
    levels_b = np.linspace(0.0, b_max_val, 60)
    contour_b = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=levels_b, cmap="jet", vmin=0.0, vmax=b_max_val)
    fig1.colorbar(contour_b, ax=axs[0, 1], label=r"$|B|$ (T)")
    axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_aspect('equal')
    
    contour_bx = axs[1, 0].contourf(X_grid, Y_grid, B_x_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_bx, ax=axs[1, 0], label=r"$B_x$ (T)")
    axs[1, 0].set_title(r"Magnetic Field Component ($B_x$)")
    axs[1, 0].set_aspect('equal')
    
    contour_by = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_by, ax=axs[1, 1], label=r"$B_y$ (T)")
    axs[1, 1].set_title(r"Magnetic Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')
    fig1.tight_layout()
    plt.savefig('figure/final_fields_grid.png', dpi=150)
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    contour_b_bg = ax2.contourf(X_grid, Y_grid, B_mag_grid, levels=levels_b, cmap="jet", alpha=0.35, vmin=0.0, vmax=b_max_val)
    fig2.colorbar(contour_b_bg, ax=ax2, label=r"$|B|$ (T)")
    
    step = 6
    B_sub = np.maximum(B_mag_grid[::step, ::step], 1e-12)
    ax2.quiver(
        X_grid[::step, ::step], Y_grid[::step, ::step],
        B_x_grid[::step, ::step] / B_sub, B_y_grid[::step, ::step] / B_sub,
        scale=35.0, color='black', pivot='mid'
    )
    ax2.set_title(r"Magnetic Flux Density Vectors ($\mathbf{B}$)")
    ax2.set_aspect('equal')
    fig2.tight_layout()
    
    plt.savefig('figure/final_results.png', dpi=150)
    plt.close('all')
    print('[*] Đã lưu kết quả mô phỏng vào figure/final_results.png và figure/final_fields_grid.png')

if __name__ == "__main__":
    main()
'''

    with open(pinn_path, 'w', encoding='utf-8') as f:
        f.write(pinn_content)
    print(f"[✓] Đã cập nhật: {pinn_path}")

    with open(visualizer_path, 'w', encoding='utf-8') as f:
        f.write(visualizer_content)
    print(f"[✓] Đã cập nhật: {visualizer_path}")

    with open(test_sim_path, 'w', encoding='utf-8') as f:
        f.write(test_sim_content)
    print(f"[✓] Đã cập nhật: {test_sim_path}")

    print("[*] Quá trình modify_project đã hoàn tất thành công.")

if __name__ == '__main__':
    modify_project()