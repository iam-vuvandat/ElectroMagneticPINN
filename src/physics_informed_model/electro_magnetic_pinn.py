import torch
from pinn_architecture import PINNArchitecture
from training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from geometry_engine.geometry import Geometry
from physics_domain.collocation_sampler import CollocationSampler

class ElectroMagneticPINN:
    def __init__(
        self, 
        geometry_config=None,
        sampler_config=None,
        pinn_config=None,
        training_config=None
    ):
        # 1. LƯU TRỮ CÁC CẤU HÌNH NHƯ LÀ THUỘC TÍNH CỦA CLASS MẸ
        self.geometry_config = geometry_config if geometry_config is not None else {}
        self.sampler_config = sampler_config if sampler_config is not None else {
            "x_boundaries": (-0.05, 0.05),
            "y_boundaries": (-0.05, 0.05)
        }
        self.pinn_config = pinn_config if pinn_config is not None else {}
        self.training_config = training_config if training_config is not None else {}
        
        # 2. KHỞI TẠO TRỰC TIẾP CÁC LỚP CON (LOẠI BỎ TIÊM PHỤ THUỘC)
        self.geometry_engine_instance = Geometry(**self.geometry_config)
        
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.get("x_boundaries", (-0.05, 0.05)),
            y_boundaries_tuple=self.sampler_config.get("y_boundaries", (-0.05, 0.05))
        )
        
        # Thiết lập hằng số quy chuẩn
        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = 795774.715459  # Hằng số chân không (vì đã bị gỡ khỏi Geometry)
        self.A0 = (self.H0 * self.L0) / self.nu0
        
        # 3. TRUYỀN CẤU HÌNH VÀO CÁC LỚP LÕI THÔNG QUA **KWARGS
        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **self.pinn_config
        )
        
        self.maxwell_pde_loss_instance = MaxwellPDELoss(L0=self.L0, H0=self.H0, nu0=self.nu0)
        
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            **self.training_config
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
            epochs=epochs_adam,
            points_tensor=points_tensor,
            reluctivity_tensor=physical_properties_dictionary["reluctivity"],
            current_density_z_tensor=physical_properties_dictionary["current_density_z"],
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"],
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )
        
        print(f"--- L-BFGS Refinement ({epochs_lbfgs} Epochs) ---")
        self.training_manager_instance.train_lbfgs(
            epochs=epochs_lbfgs, 
            points_tensor=points_tensor, 
            reluctivity_tensor=physical_properties_dictionary["reluctivity"], 
            current_density_z_tensor=physical_properties_dictionary["current_density_z"], 
            coercive_field_x_tensor=physical_properties_dictionary["coercive_field_x"], 
            coercive_field_y_tensor=physical_properties_dictionary["coercive_field_y"]
        )

    def predict_magnetic_vector_potential(self, points_tensor):
        self.pinn_architecture_instance.eval()
        with torch.no_grad():
            A_z_star = self.pinn_architecture_instance(points_tensor)
        return A_z_star * self.A0

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor.requires_grad_(True)
        
        A_z_star = self.pinn_architecture_instance(points_tensor)
        A_z_phys = A_z_star * self.A0
        
        grad_A = torch.autograd.grad(
            outputs=A_z_phys,
            inputs=points_tensor,
            grad_outputs=torch.ones_like(A_z_phys),
            create_graph=False,
            retain_graph=False
        )[0]
        
        B_x = grad_A[:, 1:2]
        B_y = -grad_A[:, 0:1]
        
        return A_z_phys.detach(), B_x.detach(), B_y.detach()
