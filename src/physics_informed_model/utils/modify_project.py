import os

def apply_stable_refactor():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))

    # =====================================================================
    # 1. CẬP NHẬT TỆP: geometry_engine/segment/segment.py (Hỗ trợ Steepness)
    # =====================================================================
    segment_file = os.path.join(project_root, 'geometry_engine', 'segment', 'segment.py')
    segment_content = '''import torch
from geometry_engine.segment.polygon_signed_distance_field import compute_polygon_signed_distance_field
from geometry_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

class Segment:
    def __init__(self, 
                 outline=None,
                 material="air",
                 relative_permeability=1.0,
                 bh_curve=None,
                 coercive=[0.0, 0.0],
                 current=0.0,
                 steepness=None):
        
        self.material = material
        self.relative_permeability = relative_permeability
        self.coercive = coercive
        self.current = current
        self.bh_curve = bh_curve
        
        self.outline = None
        self.outline_tensor = None
        self.section_area = 0.0
        self.current_density = 0.0
        
        self.steepness = 1.0 if steepness is None else steepness
        self._has_custom_steepness = steepness is not None 
        
        if outline is not None:
            self.set_outline(outline) 
            self.compute_current_density()
            if not self._has_custom_steepness:
                self.calculate_penetrating_steepness()
            
        if self.bh_curve is None:
            self.compute_constant_bh_curve()

    def compute_constant_bh_curve(self):
        constant_reluctivity = VACUUM_RELUCTIVITY / self.relative_permeability
        self.bh_curve = lambda points_tensor: torch.full((points_tensor.shape[0], 1), constant_reluctivity, dtype=torch.float32, device=points_tensor.device)
        return self.bh_curve

    def compute_section_area(self):
        if self.outline is None or len(self.outline) < 3:
            self.section_area = 0.0
            return self.section_area
        x = [p[0] for p in self.outline]
        y = [p[1] for p in self.outline]
        self.section_area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
        return self.section_area

    def compute_current_density(self):
        area = self.compute_section_area() 
        if area > 0.0:
            self.current_density = self.current / area
        else:
            self.current_density = 0.0
        return self.current_density

    def calculate_penetrating_steepness(self, grid_resolution=50):
        if self.outline_tensor is None or self.outline_tensor.shape[0] < 3:
            self.steepness = 1.0
            return self.steepness
            
        x_min = torch.min(self.outline_tensor[:, 0]).item()
        x_max = torch.max(self.outline_tensor[:, 0]).item()
        y_min = torch.min(self.outline_tensor[:, 1]).item()
        y_max = torch.max(self.outline_tensor[:, 1]).item()
        
        x_coords = torch.linspace(x_min, x_max, grid_resolution)
        y_coords = torch.linspace(y_min, y_max, grid_resolution)
        X_grid, Y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        grid_points = torch.stack([X_grid.ravel(), Y_grid.ravel()], dim=1).to(self.outline_tensor.device)
        
        sdf_values = self.compute_signed_distance_field(grid_points)
        min_sdf = torch.min(sdf_values).item() 
        
        if min_sdf >= 0.0:
            self.steepness = 0.0 
            return self.steepness
            
        d_max = abs(min_sdf)
        ideal_k = 6.0 / d_max
        s = (5000.0 - ideal_k) / 4960.0
        
        self.steepness = max(0.0, min(1.0, float(s)))
        return self.steepness

    def set_outline(self, outline):
        self.outline = outline
        self.outline_tensor = torch.tensor(outline, dtype=torch.float32)
        self.compute_section_area()
        return self

    def compute_signed_distance_field(self, points_tensor):
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(self, points_tensor):
        return self.bh_curve(points_tensor)

    def evaluate_magnetization_vector(self, points_tensor):
        hx_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[0], dtype=torch.float32, device=points_tensor.device)
        hy_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[1], dtype=torch.float32, device=points_tensor.device)
        return hx_tensor, hy_tensor

    def evaluate_current_density(self, points_tensor):
        return torch.full((points_tensor.shape[0], 1), self.current_density, dtype=torch.float32, device=points_tensor.device)
'''
    with open(segment_file, 'w', encoding='utf-8') as f:
        f.write(segment_content)
    print(f"[*] Đã cập nhật ổn định Segment tại: {segment_file}")

    # =====================================================================
    # 2. CẬP NHẬT TỆP: training_manager.py (Bảo vệ NaN & Bỏ L-BFGS)
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
        
        self.resample_frequency = resample_frequency
        self.curriculum_ratio = curriculum_ratio
        self.loss_weight_uniform = loss_weight_uniform
        self.loss_weight_interface = loss_weight_interface
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_learning_rate_adam)
        
        self.visualizer = visualizer
        self.visualizer_update_interval = visualizer_update_interval

    def compute_dynamic_loss(self, pts_u, props_u, pts_i, props_i):
        # 1. Loss Vùng Nền
        A_z_u = self.model(pts_u)
        res_u = self.pde_evaluator.compute_residual(
            xy=pts_u, A_z_star=A_z_u, nu=props_u["reluctivity"],
            J_z=props_u["current_density_z"], H_cx=props_u["coercive_field_x"], H_cy=props_u["coercive_field_y"]
        )
        loss_u = torch.mean(res_u**2) * self.loss_weight_uniform

        # 2. Loss Ranh Giới (Được kìm hãm trọng số)
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
            
            # PHÒNG THỦ: Kiểm tra NaN an toàn tuyệt đối
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1.5 * best_loss:
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
'''
    with open(manager_file, 'w', encoding='utf-8') as f:
        f.write(manager_content)
    print(f"[*] Đã tối ưu hóa ổn định TrainingManager tại: {manager_file}")

    # =====================================================================
    # 3. CẬP NHẬT TỆP: electro_magnetic_pinn.py (Loại bỏ L-BFGS khỏi luồng chạy)
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
            epochs_adam=6000,              # Tăng nhẹ số epoch Adam để bù đắp cho L-BFGS
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
            
        self.training_manager_instance = TrainingManager(
            model=self.pinn_architecture_instance,
            pde_evaluator=self.maxwell_pde_loss_instance,
            visualizer=self.visualizer_instance,
            visualizer_update_interval=self.visualization_config.update_interval,
            **manager_kwargs
        )

    def execute_training_process(self):
        print(f"\\n--- Robust Dynamic Adam Training ({self.training_config.epochs_adam} Epochs) ---")
        self.training_manager_instance.train_adam(
            epochs=self.training_config.epochs_adam,
            sampler=self.collocation_sampler_instance,
            geometry=self.geometry_engine_instance,
            sampler_config=self.sampler_config
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
    print(f"[*] Đã tối ưu hóa ổn định ElectroMagneticPINN tại: {pinn_file}")

    # =====================================================================
    # 4. CẬP NHẬT TỆP: test_simulation.py (Khai báo steepness=0.1 cho Nam châm)
    # =====================================================================
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()

    # Dùng đoạn mã thay thế sạch sẽ cho test_simulation.py
    # Đảm bảo cấu hình khai báo Segment có steepness=0.1 và bỏ qua lệnh gọi train_lbfgs cũ nếu có
    if "steepness=0.1" not in test_content:
        # Thay thế đoạn khởi tạo Segment cũ
        old_segments = """    i_magnet_left = Segment(
        outline=[[-0.04, -0.04], [-0.02, -0.04], [-0.02, 0.04], [-0.04, 0.04]],
        material="i_bar_left", 
        relative_permeability=1.05, 
        coercive=[0.0, 800000.0]
    )

    i_magnet_right = Segment(
        outline=[[0.02, -0.04], [0.04, -0.04], [0.04, 0.04], [0.02, 0.04]],
        material="i_bar_right", 
        relative_permeability=1.05, 
        coercive=[0.0, -800000.0]
    )"""

        new_segments = """    i_magnet_left = Segment(
        outline=[[-0.04, -0.04], [-0.02, -0.04], [-0.02, 0.04], [-0.04, 0.04]],
        material="i_bar_left", 
        relative_permeability=1.05, 
        coercive=[0.0, 800000.0],
        steepness=0.1
    )

    i_magnet_right = Segment(
        outline=[[0.02, -0.04], [0.04, -0.04], [0.04, 0.04], [0.02, 0.04]],
        material="i_bar_right", 
        relative_permeability=1.05, 
        coercive=[0.0, -800000.0],
        steepness=0.1
    )"""
        test_content = test_content.replace(old_segments, new_segments)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"[*] Đã cập nhật steepness cho segment trong: {test_file}")

    print("\\n[+] ĐÃ HOÀN TẤT TÁI CẤU TRÚC HỆ THỐNG ỔN ĐỊNH TUYỆT ĐỐI!")

if __name__ == "__main__":
    apply_stable_refactor()