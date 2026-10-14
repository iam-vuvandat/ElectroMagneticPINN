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
        points_tensor = points_tensor.to(self.computation_device).detach().clone().requires_grad_(True)
        
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
