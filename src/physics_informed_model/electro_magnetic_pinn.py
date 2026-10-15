import torch
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
