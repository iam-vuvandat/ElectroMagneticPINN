import torch
import torch.nn as nn
from types import SimpleNamespace
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from geometry_engine.geometry import Geometry
from physics_domain.collocation_sampler import CollocationSampler

class ElectroMagneticPINN:
    def __init__(self):
        # 1. KHỞI TẠO CẤU HÌNH TEMPLATE MẶC ĐỊNH
        self.sampler_config = SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05)
        )
        
        self.pinn_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU()
        )
        
        self.training_config = SimpleNamespace(
            lr_adam=1e-3,
            target_loss=1e-3,
            lbfgs_lr=0.8,
            lbfgs_max_iter=1000,
            lbfgs_max_eval=1250,
            lbfgs_tolerance_grad=1e-8,
            lbfgs_tolerance_change=1e-10,
            lbfgs_history_size=50
        )
        
        # Geometry độc lập với các thông số vật lý nên chỉ cần tạo 1 lần
        self.geometry_engine_instance = Geometry()
        
        # Xây dựng hệ thống lần đầu
        self._build_system()

    def update_configuration(self, sampler_config=None, pinn_config=None, training_config=None):
        # 2. HÀM UPDATE: Ghi đè cấu hình mới vào template và xây dựng lại hệ thống
        if sampler_config:
            for key, value in vars(sampler_config).items():
                setattr(self.sampler_config, key, value)
                
        if pinn_config:
            for key, value in vars(pinn_config).items():
                setattr(self.pinn_config, key, value)
                
        if training_config:
            for key, value in vars(training_config).items():
                setattr(self.training_config, key, value)
                
        # Khởi tạo lại các thành phần con dựa trên cấu hình đã cập nhật
        self._build_system()

    def _build_system(self):
        # Khởi tạo Sampler
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.x_boundaries_tuple,
            y_boundaries_tuple=self.sampler_config.y_boundaries_tuple
        )
        
        # Tính toán các hằng số quy chuẩn
        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = 795774.715459
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        # Khởi tạo Mạng Neural (truyền kwargs từ SimpleNamespace)
        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **vars(self.pinn_config)
        )
        
        # Khởi tạo Động cơ PDE
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        # Khởi tạo Training Manager
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            **vars(self.training_config)
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
            epochs=epochs_adam, points_tensor=points_tensor,
            reluctivity_tensor=physical_properties_dictionary["reluctivity"],
            current_density_z_tensor=physical_properties_dictionary["current_density_z"],
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"],
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )
        
        print(f"--- L-BFGS Refinement ({epochs_lbfgs} Epochs) ---")
        self.training_manager_instance.train_lbfgs(
            epochs=epochs_lbfgs, points_tensor=points_tensor,
            reluctivity_tensor=physical_properties_dictionary["reluctivity"],
            current_density_z_tensor=physical_properties_dictionary["current_density_z"],
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"],
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor.requires_grad_(True)
        A_z_star = self.pinn_architecture_instance(points_tensor)
        A_z_phys = A_z_star * self.A0
        
        grad_A = torch.autograd.grad(
            outputs=A_z_phys, inputs=points_tensor,
            grad_outputs=torch.ones_like(A_z_phys), create_graph=False
        )[0]
        
        return A_z_phys.detach(), grad_A[:, 1:2].detach(), (-grad_A[:, 0:1]).detach()
