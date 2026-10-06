import os

def refactor_to_dynamic_curriculum():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # =====================================================================
    # TỆP 1: TÁI CẤU TRÚC LỚP BỌC (electro_magnetic_pinn.py)
    # =====================================================================
    pinn_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    pinn_content = '''import torch
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
        print(f"\\n--- [GIAI ĐOẠN 1] Dynamic Curriculum Adam ({self.training_config.epochs_adam} Epochs) ---")
        self.training_manager_instance.train_adam(
            epochs=self.training_config.epochs_adam,
            sampler=self.collocation_sampler_instance,
            geometry=self.geometry_engine_instance,
            sampler_config=self.sampler_config
        )
        
        print(f"\\n--- [GIAI ĐOẠN 2] Full-Batch L-BFGS Refinement ({self.training_config.epochs_lbfgs} Epochs) ---")
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
'''
    with open(pinn_file, 'w', encoding='utf-8') as f:
        f.write(pinn_content)
    print(f"[*] Đã tái cấu trúc: {pinn_file}")

    # =====================================================================
    # TỆP 2: TÁI CẤU TRÚC BỘ QUẢN LÝ CHIẾN THUẬT (training_manager.py)
    # =====================================================================
    manager_file = os.path.join(project_root, 'training_manager.py')
    manager_content = '''import torch
import torch.optim as optim

class TrainingManager:
    def __init__(
        self, 
        model, 
        pde_evaluator, 
        learning_rate_adam=1e-3, 
        target_loss=0.0,
        lbfgs_learning_rate=0.8, 
        lbfgs_maximum_iterations=1000, 
        lbfgs_maximum_evaluations=1250,
        lbfgs_tolerance_gradient=1e-8,
        lbfgs_tolerance_change=1e-10,
        lbfgs_history_size=50,
        visualizer=None,
        visualizer_update_interval=100,
        resample_frequency=500,
        curriculum_ratio=0.25,
        loss_weight_uniform=1.0,
        loss_weight_interface=0.1
    ):
        self.model = model
        self.pde_evaluator = pde_evaluator
        self.target_loss = target_loss
        self.base_learning_rate_adam = learning_rate_adam
        
        # BỔ SUNG: Tham số chiến thuật động
        self.resample_frequency = resample_frequency
        self.curriculum_ratio = curriculum_ratio
        self.loss_weight_uniform = loss_weight_uniform
        self.loss_weight_interface = loss_weight_interface
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=lbfgs_learning_rate,
            max_iter=lbfgs_maximum_iterations,
            max_eval=lbfgs_maximum_evaluations,
            tolerance_grad=lbfgs_tolerance_gradient,
            tolerance_change=lbfgs_tolerance_change,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )
        
        self.visualizer = visualizer
        self.visualizer_update_interval = visualizer_update_interval

    def compute_dynamic_loss(self, pts_u, props_u, pts_i, props_i):
        # 1. Tính Loss Vùng Nền (Uniform)
        A_z_u = self.model(pts_u)
        res_u = self.pde_evaluator.compute_residual(
            xy=pts_u, A_z_star=A_z_u, nu=props_u["reluctivity"],
            J_z=props_u["current_density_z"], H_cx=props_u["coercive_field_x"], H_cy=props_u["coercive_field_y"]
        )
        loss_u = torch.mean(res_u**2) * self.loss_weight_uniform

        # 2. Tính Loss Ranh Giới (Interface)
        A_z_i = self.model(pts_i)
        res_i = self.pde_evaluator.compute_residual(
            xy=pts_i, A_z_star=A_z_i, nu=props_i["reluctivity"],
            J_z=props_i["current_density_z"], H_cx=props_i["coercive_field_x"], H_cy=props_i["coercive_field_y"]
        )
        loss_i = torch.mean(res_i**2) * self.loss_weight_interface

        return loss_u + loss_i

    def _trigger_visualization(self, epoch, loss_val, pts_u, props_u, pts_i, props_i):
        if self.visualizer:
            comb_pts = torch.cat([pts_u, pts_i], dim=0)
            comb_nu = torch.cat([props_u['reluctivity'], props_i['reluctivity']], dim=0)
            comb_jz = torch.cat([props_u['current_density_z'], props_i['current_density_z']], dim=0)
            comb_hcx = torch.cat([props_u['coercive_field_x'], props_i['coercive_field_x']], dim=0)
            comb_hcy = torch.cat([props_u['coercive_field_y'], props_i['coercive_field_y']], dim=0)
            self.visualizer.save_frame(epoch, loss_val, comb_pts, comb_nu, comb_jz, comb_hcx, comb_hcy)

    def train_adam(self, epochs, sampler, geometry, sampler_config):
        self.model.train()
        best_loss = float('inf')
        best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
        
        num_u = int(sampler_config.number_of_uniform_points * self.curriculum_ratio)
        num_i = int(sampler_config.number_of_interface_points * self.curriculum_ratio)
        threshold = sampler_config.distance_threshold
        
        pts_u, props_u, pts_i, props_i = None, None, None, None
        
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(epochs):
            if epoch == 0 or (self.resample_frequency > 0 and epoch % self.resample_frequency == 0):
                pts_u = sampler.generate_uniform_points_tensor(num_u)
                pts_i = sampler.generate_interface_points_tensor(geometry, num_i, threshold)
                props_u = geometry.evaluate_global_physical_properties(pts_u)
                props_i = geometry.evaluate_global_physical_properties(pts_i)

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_dynamic_loss(pts_u, props_u, pts_i, props_i)
            
            if torch.isnan(loss) or loss.item() > 1.5 * best_loss:
                self.model.load_state_dict(best_model_state)
                for param_group in self.optimizer_adam.param_groups:
                    param_group['lr'] *= 0.8
                continue
                
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            current_loss_value = loss.item()
            if current_loss_value < best_loss:
                best_loss = current_loss_value
                best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                
            if self.target_loss > 0 and current_loss_value <= self.target_loss:
                print(f"Adam Epoch {epoch + 1}: Đạt target_loss. KẾT THÚC SỚM!")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Adam Epoch {epoch + 1}: Loss = {current_loss_value:.6e} | LR = {self.optimizer_adam.param_groups[0]['lr']:.3e}")
                
            if (epoch + 1) % self.visualizer_update_interval == 0:
                self._trigger_visualization(epoch + 1, current_loss_value, pts_u, props_u, pts_i, props_i)

    def train_lbfgs(self, epochs, pts_u, props_u, pts_i, props_i):
        self.model.train()
        lbfgs_counter = [0]
        early_stop_triggered = False 
        
        def closure():
            nonlocal early_stop_triggered
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_dynamic_loss(pts_u, props_u, pts_i, props_i)
            loss.backward(retain_graph=True)
            
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
                
            if lbfgs_counter[0] % self.visualizer_update_interval == 0:
                self._trigger_visualization(50000 + lbfgs_counter[0], loss.item(), pts_u, props_u, pts_i, props_i)
                
            if self.target_loss > 0 and loss.item() <= self.target_loss:
                early_stop_triggered = True
            return loss
            
        for epoch in range(epochs):
            if early_stop_triggered:
                print(f"L-BFGS Epoch {epoch + 1}: Đạt target_loss. KẾT THÚC SỚM!")
                break
            self.optimizer_lbfgs.step(closure)
'''
    with open(manager_file, 'w', encoding='utf-8') as f:
        f.write(manager_content)
    print(f"[*] Đã tái cấu trúc: {manager_file}")
    print("\\n[+] ĐÃ HOÀN TẤT ĐỢT BIG REFACTOR TỐI ƯU HÓA ĐỘNG!")

if __name__ == "__main__":
    refactor_to_dynamic_curriculum()