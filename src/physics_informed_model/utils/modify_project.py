import os

def apply_full_visualization_encapsulation():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    root_pinn_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    test_file = os.path.join(project_root, 'tests', 'test_simulation.py')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: electro_magnetic_pinn.py (Tích hợp phương thức plot_fields)
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
        # 1. Cấu hình Lấy mẫu điểm (Collocation Sampler Configuration)
        self.sampler_config = SimpleNamespace(
            dynamic_resampling=True,
            resample_interval=100,
            resample_points_before_lbfgs_stage=True
        )
        
        # 2. Cấu hình Tiến trình Huấn luyện (Training Process Configuration)
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
        
        # 3. Cấu hình Trực quan hóa (Visualization Configuration)
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
        """
        Giao diện suy luận cấp cao (High-level Inference API):
        Tính toán trường Az, Bx, By trên toàn miền vật lý từ các mạng con XPINN.
        """
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
        """
        Đóng gói toàn bộ logic khởi tạo lưới 2D, suy luận trường và hiển thị/lưu đồ thị 4 ô (Az, |B|, mu_r, By).
        """
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
        
        if not (np.all(np.isfinite(A_z_np)) and np.all(np.isfinite(B_x_np))):
            raise ValueError("Lỗi: Trường từ trường dự đoán chứa giá trị NaN hoặc Inf!")

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

        # GIAI ĐOẠN 1A: ADAM LINEAR WARM-UP
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

        # GIAI ĐOẠN 1B: ADAM NON-LINEAR ADAPTATION
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

        # GIAI ĐOẠN 2: L-BFGS REFINEMENT
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

    # =========================================================================
    # 2. CẬP NHẬT: tests/test_simulation.py (Siêu tinh gọn với 1 dòng plot_fields)
    # =========================================================================
    files_to_update[test_file] = '''import os
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

def run_modern_simulation_test():
    output_dir = "figure_modern_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN với API Cấu hình mới (Linear Warm-up Only)...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. Cấu hình chuẩn hóa trường vật lý
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    # 2. Cấu hình Lấy mẫu động
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # 3. Cấu hình Bật/Tắt Giai đoạn & Thuật toán Tối ưu
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = False
    model.training_config.enable_lbfgs_refinement_stage = False

    model.training_config.number_of_adam_epochs = 1000
    model.training_config.linear_warmup_epochs_ratio = 1.0
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-7
    model.training_config.number_of_lbfgs_iterations = 0

    # 4. Cấu hình Trực quan hóa
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "modern_training.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "modern_training.mp4"

    def apply_standard_config(domain, int_pts=2000, surf_pts=1200, pde_nonlin_w=1.0):
        domain.training_config.pde_weight = 1.0
        domain.training_config.pde_nonlinear_weight = pde_nonlin_w
        domain.training_config.interface_weight = 1.0
        domain.training_config.interface_potential_weight = 1000.0
        domain.training_config.interface_bn_weight = 400.0
        domain.training_config.interface_ht_weight = 400.0
        domain.training_config.pde_fading_beta = 80.0
        domain.training_config.number_of_interior_points = int_pts
        domain.training_config.number_of_surface_points = surf_pts
        
        domain.network_config.hidden_layers = 4
        domain.network_config.hidden_neurons = 64
        domain.network_config.activation_function = nn.SiLU()
        domain.network_config.fourier_features = True
        domain.network_config.fourier_sigma = 1.2
        domain.network_config.fourier_mapping_size = 128
        return domain.build_domain()

    # 1. Khung Ách sắt (Iron Yoke - steel_1008)
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name="steel_1008",
        outline=[
            [-0.06, -0.03], 
            [0.06, -0.03], 
            [0.06, 0.0], 
            [-0.06, 0.0]
        ]
    )
    apply_standard_config(iron_yoke, int_pts=3000, surf_pts=1600, pde_nonlin_w=2.0)

    # 2. Nam châm trái (N30UH)
    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left",
        material_name="N30UH",
        outline=[
            [-0.05, 0.0], 
            [-0.005, 0.0], 
            [-0.005, 0.018], 
            [-0.05, 0.018]
        ],
        coercive_direction=(0, 1)
    )
    apply_standard_config(magnet_left, int_pts=2000, surf_pts=1200, pde_nonlin_w=1.0)

    # 3. Nam châm phải (N30UH)
    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right",
        material_name="N30UH",
        outline=[
            [0.005, 0.0], 
            [0.05, 0.0], 
            [0.05, 0.018], 
            [0.005, 0.018]
        ],
        coercive_direction=(0, -1)
    )
    apply_standard_config(magnet_right, int_pts=2000, surf_pts=1200, pde_nonlin_w=1.0)

    # 4. Background Air
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        pde_nonlinear_weight=1.0,
        interface_weight=1.0,
        pde_fading_beta=80.0,
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
    sim_domain.plot_problem_definition(resolution=100)
    
    print("[*] Chạy tiến trình huấn luyện kiểm thử...")
    model.execute_training_process()

    print("[*] Đang đánh giá và xuất biểu đồ từ trường bằng High-Level API...")
    model.plot_fields(
        resolution=180, 
        save_path=f"{output_dir}/modern_final_fields.png"
    )

    print(f"[✔] HOÀN THÀNH BÀI KIỂM THỬ TRONG 1 DÒNG LỆNH VẼ BIỂU ĐỒ! Kết quả lưu tại {output_dir}/")

def test_simulation4():
    run_modern_simulation_test()

if __name__ == "__main__":
    run_modern_simulation_test()
'''

    for file_path, code_content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        print(f"[*] Đã ghi và cập nhật thành công: {os.path.relpath(file_path, project_root)}")

    print("\n[✔] HOÀN THÀNH ĐÓNG GÓI NỐT TOÀN BỘ LOGIC VẼ ĐỒ THỊ VÀO CLASS MẸ!")

if __name__ == "__main__":
    apply_full_visualization_encapsulation()