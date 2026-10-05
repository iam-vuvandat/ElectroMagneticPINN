import os

def execute_gpu_and_vram_optimization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # ---------------------------------------------------------
    # 1. TỐI ƯU HÓA SAMPLER (Sinh điểm Natively trên GPU)
    # ---------------------------------------------------------
    sampler_file = os.path.join(project_root, 'physics_domain', 'collocation_sampler.py')
    sampler_code = """import torch

class CollocationSampler:
    def __init__(self, x_boundaries_tuple, y_boundaries_tuple, device=torch.device('cpu')):
        self.x_minimum = x_boundaries_tuple[0]
        self.x_maximum = x_boundaries_tuple[1]
        self.y_minimum = y_boundaries_tuple[0]
        self.y_maximum = y_boundaries_tuple[1]
        self.device = device

    def generate_uniform_points_tensor(self, number_of_points):
        # Sinh trực tiếp trên GPU
        points_tensor = torch.rand((number_of_points, 2), dtype=torch.float32, device=self.device)
        points_tensor[:, 0] = points_tensor[:, 0] * (self.x_maximum - self.x_minimum) + self.x_minimum
        points_tensor[:, 1] = points_tensor[:, 1] * (self.y_maximum - self.y_minimum) + self.y_minimum
        points_tensor.requires_grad_(True)
        return points_tensor

    def generate_interface_points_tensor(self, geometry_object, number_of_points, distance_threshold):
        collected_points = []
        collected_count = 0
        pool_size_value = number_of_points * 20
        
        while collected_count < number_of_points:
            # Quét ranh giới siêu tốc bằng GPU
            points_pool_tensor = torch.rand((pool_size_value, 2), dtype=torch.float32, device=self.device)
            points_pool_tensor[:, 0] = points_pool_tensor[:, 0] * (self.x_maximum - self.x_minimum) + self.x_minimum
            points_pool_tensor[:, 1] = points_pool_tensor[:, 1] * (self.y_maximum - self.y_minimum) + self.y_minimum
            
            signed_distance_field_tensor = geometry_object.compute_global_signed_distance_field(points_pool_tensor)
            mask_tensor = torch.abs(signed_distance_field_tensor) < distance_threshold
            mask_1d = mask_tensor.squeeze()
            
            if mask_1d.any():
                valid_points = points_pool_tensor[mask_1d]
                collected_points.append(valid_points)
                collected_count += valid_points.shape[0]
                
        interface_points_tensor = torch.cat(collected_points, dim=0)[:number_of_points, :]
        interface_points_tensor = interface_points_tensor.detach().clone()
        interface_points_tensor.requires_grad_(True)
        return interface_points_tensor

    def generate_combined_points_tensor(self, geometry_object, number_of_uniform_points, number_of_interface_points, distance_threshold):
        uniform_points_tensor = self.generate_uniform_points_tensor(number_of_uniform_points)
        interface_points_tensor = self.generate_interface_points_tensor(geometry_object, number_of_interface_points, distance_threshold)
        
        combined_points_tensor = torch.cat([uniform_points_tensor, interface_points_tensor], dim=0)
        
        combined_points_tensor = combined_points_tensor.detach().clone()
        combined_points_tensor.requires_grad_(True)
        return combined_points_tensor
"""

    # ---------------------------------------------------------
    # 2. TỐI ƯU HÓA LỚP BỌC CHÍNH (Gán Device cho hệ thống)
    # ---------------------------------------------------------
    emp_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    with open(emp_file, 'r', encoding='utf-8') as f:
        emp_content = f.read()
    
    old_sampler_init = """        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.x_boundaries_tuple,
            y_boundaries_tuple=self.sampler_config.y_boundaries_tuple
        )"""
    new_sampler_init = """        self.computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Kích hoạt phần cứng: {self.computation_device}")

        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.x_boundaries_tuple,
            y_boundaries_tuple=self.sampler_config.y_boundaries_tuple,
            device=self.computation_device
        )"""
    emp_content = emp_content.replace(old_sampler_init, new_sampler_init)
    
    old_pinn_init = """        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **vars(self.pinn_config)
        )"""
    new_pinn_init = """        self.pinn_architecture_instance = PINNArchitecture(
            domain_scale=self.L0,
            **vars(self.pinn_config)
        ).to(self.computation_device)"""
    emp_content = emp_content.replace(old_pinn_init, new_pinn_init)

    old_eval_fields = """    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor.requires_grad_(True)"""
    new_eval_fields = """    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor = points_tensor.to(self.computation_device).clone().requires_grad_(True)"""
    emp_content = emp_content.replace(old_eval_fields, new_eval_fields)

    # ---------------------------------------------------------
    # 3. TỐI ƯU HÓA BỘ NHỚ VRAM (Xóa retain_graph=True)
    # ---------------------------------------------------------
    training_file = os.path.join(project_root, 'training_manager.py')
    with open(training_file, 'r', encoding='utf-8') as f:
        training_content = f.read()
    # Chữa lỗi Adam
    training_content = training_content.replace("loss.backward(retain_graph=True)", "loss.backward()")

    pde_file = os.path.join(project_root, 'physics_domain', 'physical_equations', 'maxwell_pde_loss.py')
    with open(pde_file, 'r', encoding='utf-8') as f:
        pde_content = f.read()
    
    old_grad_hy = """        grad_Hy_star = torch.autograd.grad(
            outputs=H_y_star,
            inputs=xy,
            grad_outputs=torch.ones_like(H_y_star),
            create_graph=True,
            retain_graph=True
        )[0]"""
    new_grad_hy = """        grad_Hy_star = torch.autograd.grad(
            outputs=H_y_star,
            inputs=xy,
            grad_outputs=torch.ones_like(H_y_star),
            create_graph=True,
            retain_graph=False
        )[0]"""
    pde_content = pde_content.replace(old_grad_hy, new_grad_hy)

    # Ghi lại toàn bộ các file
    with open(sampler_file, 'w', encoding='utf-8') as f: f.write(sampler_code)
    with open(emp_file, 'w', encoding='utf-8') as f: f.write(emp_content)
    with open(training_file, 'w', encoding='utf-8') as f: f.write(training_content)
    with open(pde_file, 'w', encoding='utf-8') as f: f.write(pde_content)

    print("Hoàn tất tối ưu hóa GPU & VRAM! Hiệu năng hệ thống sẽ tăng vọt.")

if __name__ == "__main__":
    execute_gpu_and_vram_optimization()