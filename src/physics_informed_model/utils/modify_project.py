import os

def execute_ultimate_fix():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # =====================================================================
    # 1. COLLOCATION SAMPLER (Khởi tạo điểm Natively trên GPU)
    # =====================================================================
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

    # =====================================================================
    # 2. TRAINING MANAGER (Cô lập Đồ thị & Chống tràn VRAM)
    # =====================================================================
    training_file = os.path.join(project_root, 'training_manager.py')
    with open(training_file, 'r', encoding='utf-8') as f:
        training_content = f.read()
    
    # Vá hàm compute_loss để tạo bản sao đồ thị độc lập
    old_compute = """    def compute_loss(self, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        A_z_star = self.model(points_tensor)
        residual_star = self.pde_evaluator.compute_residual(
            xy=points_tensor, A_z_star=A_z_star, nu=reluctivity_tensor,"""
    new_compute = """    def compute_loss(self, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        xy = points_tensor.detach().clone().requires_grad_(True)
        A_z_star = self.model(xy)
        residual_star = self.pde_evaluator.compute_residual(
            xy=xy, A_z_star=A_z_star, nu=reluctivity_tensor,"""
    training_content = training_content.replace(old_compute, new_compute)
    # Gỡ bỏ hoàn toàn retain_graph=True
    training_content = training_content.replace("loss.backward(retain_graph=True)", "loss.backward()")

    # =====================================================================
    # 3. GEOMETRY ENGINE (Khôi phục CSG Unite + Softmax)
    # =====================================================================
    geometry_file = os.path.join(project_root, 'geometry_engine', 'geometry.py')
    geometry_code = """import torch
from geometry_engine.segment.segment import Segment
from geometry_engine.global_signed_distance_field import compute_global_signed_distance_field
from geometry_engine.global_physical_properties_evaluation import evaluate_global_physical_properties
from geometry_engine.geometry_visualizer import plot_geometry_problem

class Geometry:
    def __init__(self):
        self.segments_list = []

    def add_segment(self, segment_object):
        self.segments_list.append(segment_object)
        return self

    def unite(self, segments_collection):
        united_segment = Segment()
        united_segment.components = []
        for item in segments_collection:
            if isinstance(item, Segment):
                united_segment.components.append(item)
            else:
                united_segment.components.append(Segment(outline=item))

        def compute_united_sdf(points_tensor):
            if not united_segment.components:
                return torch.ones((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            res_sdf = united_segment.components[0].compute_signed_distance_field(points_tensor)
            for comp in united_segment.components[1:]:
                res_sdf = torch.minimum(res_sdf, comp.compute_signed_distance_field(points_tensor))
            return res_sdf
        united_segment.compute_signed_distance_field = compute_united_sdf

        def evaluate_united_reluctivity(points_tensor):
            sdfs = [comp.compute_signed_distance_field(points_tensor) for comp in united_segment.components]
            stacked_sdfs = torch.stack(sdfs, dim=1).squeeze(-1)
            weights = torch.softmax(-100.0 * stacked_sdfs, dim=1)
            blended_nu = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            for i, comp in enumerate(united_segment.components):
                blended_nu += weights[:, i:i+1] * comp.evaluate_reluctivity(points_tensor)
            return blended_nu
        united_segment.evaluate_reluctivity = evaluate_united_reluctivity

        def evaluate_united_magnetization(points_tensor):
            sdfs = [comp.compute_signed_distance_field(points_tensor) for comp in united_segment.components]
            stacked_sdfs = torch.stack(sdfs, dim=1).squeeze(-1)
            weights = torch.softmax(-100.0 * stacked_sdfs, dim=1)
            blended_hx = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            blended_hy = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            for i, comp in enumerate(united_segment.components):
                hx, hy = comp.evaluate_magnetization_vector(points_tensor)
                blended_hx += weights[:, i:i+1] * hx
                blended_hy += weights[:, i:i+1] * hy
            return blended_hx, blended_hy
        united_segment.evaluate_magnetization_vector = evaluate_united_magnetization

        def evaluate_united_current_density(points_tensor):
            sdfs = [comp.compute_signed_distance_field(points_tensor) for comp in united_segment.components]
            stacked_sdfs = torch.stack(sdfs, dim=1).squeeze(-1)
            weights = torch.softmax(-100.0 * stacked_sdfs, dim=1)
            blended_jz = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            for i, comp in enumerate(united_segment.components):
                blended_jz += weights[:, i:i+1] * comp.evaluate_current_density(points_tensor)
            return blended_jz
        united_segment.evaluate_current_density = evaluate_united_current_density
        united_segment.steepness = max([comp.steepness for comp in united_segment.components]) if united_segment.components else 1.0

        self.segments_list.append(united_segment)
        return self

    def compute_global_signed_distance_field(self, points_tensor):
        return compute_global_signed_distance_field(self.segments_list, points_tensor)

    def evaluate_global_physical_properties(self, points_tensor):
        return evaluate_global_physical_properties(self.segments_list, points_tensor)

    def plot_problem_definition(self, x_boundaries_tuple, y_boundaries_tuple, resolution=100):
        plot_geometry_problem(self, x_boundaries_tuple, y_boundaries_tuple, resolution)
"""

    # =====================================================================
    # 4. ELECTRO MAGNETIC PINN (Kích hoạt Device)
    # =====================================================================
    emp_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    with open(emp_file, 'r', encoding='utf-8') as f:
        emp_content = f.read()
    
    old_sampler_init = """        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.x_boundaries_tuple,
            y_boundaries_tuple=self.sampler_config.y_boundaries_tuple
        )"""
    new_sampler_init = """        self.computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Hệ thống chạy trên: {self.computation_device}")
        self.collocation_sampler_instance = CollocationSampler(
            x_boundaries_tuple=self.sampler_config.x_boundaries_tuple,
            y_boundaries_tuple=self.sampler_config.y_boundaries_tuple,
            device=self.computation_device
        )"""
    emp_content = emp_content.replace(old_sampler_init, new_sampler_init)
    
    emp_content = emp_content.replace(
        "**vars(self.pinn_config)\n        )", 
        "**vars(self.pinn_config)\n        ).to(self.computation_device)"
    )
    emp_content = emp_content.replace(
        "points_tensor.requires_grad_(True)", 
        "points_tensor = points_tensor.to(self.computation_device).clone().requires_grad_(True)"
    )

    # =====================================================================
    # 5. MAXWELL PDE LOSS (Gỡ retain_graph cuối)
    # =====================================================================
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

    # =====================================================================
    # 6. TEST SIMULATION (Gỡ plt.show + Sử dụng Unite)
    # =====================================================================
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()
    
    # Thêm Agg backend
    test_content = test_content.replace("import matplotlib.pyplot as plt", "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt")
    
    # Đổi add_segment thành unite
    old_add = """    model.geometry_engine_instance.add_segment(left_leg)
    model.geometry_engine_instance.add_segment(yoke)
    model.geometry_engine_instance.add_segment(right_leg)"""
    test_content = test_content.replace(old_add, "    model.geometry_engine_instance.unite([left_leg, yoke, right_leg])")
    
    # Đổi plt.show() thành plt.savefig()
    test_content = test_content.replace("plt.show()", "plt.savefig('final_results.png', dpi=150)\n    plt.close('all')\n    print('[*] Đã lưu kết quả vào final_results.png')")

    # =====================================================================
    # GHI ĐÈ TOÀN BỘ FILE
    # =====================================================================
    with open(sampler_file, 'w', encoding='utf-8') as f: f.write(sampler_code)
    with open(training_file, 'w', encoding='utf-8') as f: f.write(training_content)
    with open(geometry_file, 'w', encoding='utf-8') as f: f.write(geometry_code)
    with open(emp_file, 'w', encoding='utf-8') as f: f.write(emp_content)
    with open(pde_file, 'w', encoding='utf-8') as f: f.write(pde_content)
    with open(test_file, 'w', encoding='utf-8') as f: f.write(test_content)

    print("HOÀN TẤT VÁ LỖI! Hãy chạy `python test_simulation.py` để thưởng thức sức mạnh của GPU.")

if __name__ == "__main__":
    execute_ultimate_fix()