import torch
from types import SimpleNamespace
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from geometry_engine.geometry import Geometry
from physics_domain.collocation_sampler import CollocationSampler

class ElectroMagneticPINN:
    def __init__(
        self, 
        geometry_engine_configuration=None,
        collocation_sampler_configuration=None,
        pinn_architecture_configuration=None,
        training_manager_configuration=None
    ):
        # 1. LƯU TRỮ CẤU HÌNH BẰNG SIMPLENAMESPACE (Bảo vệ giá trị mặc định)
        self.geometry_engine_configuration = geometry_engine_configuration or SimpleNamespace()
        
        self.collocation_sampler_configuration = collocation_sampler_configuration or SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05)
        )
        
        self.pinn_architecture_configuration = pinn_architecture_configuration or SimpleNamespace()
        
        self.training_manager_configuration = training_manager_configuration or SimpleNamespace()
        
        # 2. KHỞI TẠO CÁC LỚP CON (LOẠI BỎ HOÀN TOÀN TIÊM PHỤ THUỘC)
        self.geometry_engine_instance = Geometry() # Geometry hiện không nhận args
        
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.collocation_sampler_configuration.x_boundaries_tuple,
            y_boundaries_tuple=self.collocation_sampler_configuration.y_boundaries_tuple
        )
        
        # Hằng số vật lý quy chuẩn
        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = 795774.715459
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        # 3. TRUYỀN CẤU HÌNH VÀO CÁC LỚP LÕI (Sử dụng vars() để bung Namespace thành kwargs)
        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **vars(self.pinn_architecture_configuration)
        )
        
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            **vars(self.training_manager_configuration)
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
