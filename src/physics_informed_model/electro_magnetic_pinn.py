import torch
import torch.nn as nn
from types import SimpleNamespace
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from geometry_engine.geometry import Geometry
from physics_domain.collocation_sampler import CollocationSampler
from geometry_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY
from utils.training_visualizer import TrainingVisualizer

class ElectroMagneticPINN:
    def __init__(self):
        self.sampler_config = SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05),
            number_of_uniform_points=15000,
            number_of_interface_points=5000,
            distance_threshold=0.005
        )
        
        self.pinn_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU()
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
            lbfgs_history_size=50,
            
            # --- CẤU HÌNH CHIẾN THUẬT ĐỘNG (BỔ SUNG) ---
            resample_frequency=500,        
            curriculum_ratio=0.25,         
            loss_weight_uniform=1.0,       
            loss_weight_interface=0.1      
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
        
        self.geometry_engine_instance = Geometry()
        self._build_system()

    def update_electromagnetic_pinn(self):
        self._build_system()

    def _build_system(self):
        self.computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Hệ thống vật lý khởi tạo trên: {self.computation_device}")
        
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.x_boundaries_tuple,
            y_boundaries_tuple=self.sampler_config.y_boundaries_tuple,
            device=self.computation_device
        )
        
        self.L0 = self.physics_config.scale_L0
        self.H0 = self.physics_config.scale_H0
        self.nu0 = self.physics_config.scale_nu0
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **vars(self.pinn_config)
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
        print(f"\n--- [GIAI ĐOẠN 1] Dynamic Curriculum Adam ({self.training_config.epochs_adam} Epochs) ---")
        self.training_manager_instance.train_adam(
            epochs=self.training_config.epochs_adam,
            sampler=self.collocation_sampler_instance,
            geometry=self.geometry_engine_instance,
            sampler_config=self.sampler_config
        )
        
        print(f"\n--- [GIAI ĐOẠN 2] Full-Batch L-BFGS Refinement ({self.training_config.epochs_lbfgs} Epochs) ---")
        pts_u = self.collocation_sampler_instance.generate_uniform_points_tensor(
            self.sampler_config.number_of_uniform_points
        )
        pts_i = self.collocation_sampler_instance.generate_interface_points_tensor(
            self.geometry_engine_instance,
            self.sampler_config.number_of_interface_points,
            self.sampler_config.distance_threshold
        )
        
        props_u = self.geometry_engine_instance.evaluate_global_physical_properties(pts_u)
        props_i = self.geometry_engine_instance.evaluate_global_physical_properties(pts_i)

        self.training_manager_instance.train_lbfgs(
            epochs=self.training_config.epochs_lbfgs,
            pts_u=pts_u, props_u=props_u,
            pts_i=pts_i, props_i=props_i
        )
        
        if self.visualizer_instance:
            print("Đang xuất file GIF Animation...")
            self.visualizer_instance.generate_gif()

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor = points_tensor.to(self.computation_device).clone().requires_grad_(True)
        A_z_star = self.pinn_architecture_instance(points_tensor)
        A_z_phys = A_z_star * self.A0
        
        grad_A = torch.autograd.grad(
            outputs=A_z_phys, inputs=points_tensor,
            grad_outputs=torch.ones_like(A_z_phys), create_graph=False
        )[0]
        
        return A_z_phys.detach(), grad_A[:, 1:2].detach(), (-grad_A[:, 0:1]).detach()
