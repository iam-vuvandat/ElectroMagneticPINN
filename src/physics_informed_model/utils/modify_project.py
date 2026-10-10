import os

def update_point_sampling_strategy():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    sampler_file = os.path.join(project_root, 'physics_domain', 'collocation_sampler.py')
    with open(sampler_file, 'r', encoding='utf-8') as f:
        sampler_content = f.read()

    old_sampler_block = """    def generate_interface_points_tensor(self, geometry_object, number_of_points, distance_threshold):
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
        return combined_points_tensor"""

    new_sampler_block = """    def generate_material_points_tensor(self, geometry_object, number_of_points, distance_threshold):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device).requires_grad_(True)
            
        collected_points = []
        collected_count = 0
        pool_size_value = max(number_of_points * 20, 1000)
        
        while collected_count < number_of_points:
            points_pool_tensor = torch.rand((pool_size_value, 2), dtype=torch.float32, device=self.device)
            points_pool_tensor[:, 0] = points_pool_tensor[:, 0] * (self.x_maximum - self.x_minimum) + self.x_minimum
            points_pool_tensor[:, 1] = points_pool_tensor[:, 1] * (self.y_maximum - self.y_minimum) + self.y_minimum
            
            signed_distance_field_tensor = geometry_object.compute_global_signed_distance_field(points_pool_tensor)
            mask_tensor = signed_distance_field_tensor < distance_threshold
            mask_1d = mask_tensor.squeeze()
            
            if mask_1d.any():
                valid_points = points_pool_tensor[mask_1d]
                collected_points.append(valid_points)
                collected_count += valid_points.shape[0]
                
        material_points_tensor = torch.cat(collected_points, dim=0)[:number_of_points, :]
        material_points_tensor = material_points_tensor.detach().clone()
        material_points_tensor.requires_grad_(True)
        return material_points_tensor

    def generate_combined_points_tensor(self, geometry_object, number_of_uniform_points, number_of_material_points, distance_threshold):
        tensors_to_combine = []
        
        if number_of_uniform_points > 0:
            tensors_to_combine.append(self.generate_uniform_points_tensor(number_of_uniform_points))
            
        if number_of_material_points > 0:
            tensors_to_combine.append(self.generate_material_points_tensor(geometry_object, number_of_material_points, distance_threshold))
            
        combined_points_tensor = torch.cat(tensors_to_combine, dim=0)
        
        combined_points_tensor = combined_points_tensor.detach().clone()
        combined_points_tensor.requires_grad_(True)
        return combined_points_tensor"""

    if "def generate_interface_points_tensor" in sampler_content:
        sampler_content = sampler_content.replace(old_sampler_block, new_sampler_block)
        with open(sampler_file, 'w', encoding='utf-8') as f:
            f.write(sampler_content)
        print(f"[*] Đã cập nhật collocation_sampler.py")

    pinn_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    with open(pinn_file, 'r', encoding='utf-8') as f:
        pinn_content = f.read()

    old_pinn_config = """        self.sampler_config = SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05),
            number_of_uniform_points=15000,
            number_of_interface_points=5000,
            distance_threshold=0.005
        )"""

    new_pinn_config = """        self.sampler_config = SimpleNamespace(
            x_boundaries_tuple=(-0.05, 0.05),
            y_boundaries_tuple=(-0.05, 0.05),
            number_of_uniform_points=8000,
            number_of_material_points=7000,
            distance_threshold=0.005
        )"""

    old_pinn_execute = """        points_tensor = self.collocation_sampler_instance.generate_combined_points_tensor(
            geometry_object=self.geometry_engine_instance,
            number_of_uniform_points=self.sampler_config.number_of_uniform_points,
            number_of_interface_points=self.sampler_config.number_of_interface_points,
            distance_threshold=self.sampler_config.distance_threshold
        )"""

    new_pinn_execute = """        points_tensor = self.collocation_sampler_instance.generate_combined_points_tensor(
            geometry_object=self.geometry_engine_instance,
            number_of_uniform_points=self.sampler_config.number_of_uniform_points,
            number_of_material_points=self.sampler_config.number_of_material_points,
            distance_threshold=self.sampler_config.distance_threshold
        )"""

    if "number_of_interface_points" in pinn_content:
        pinn_content = pinn_content.replace(old_pinn_config, new_pinn_config)
        pinn_content = pinn_content.replace(old_pinn_execute, new_pinn_execute)
        with open(pinn_file, 'w', encoding='utf-8') as f:
            f.write(pinn_content)
        print(f"[*] Đã cập nhật electro_magnetic_pinn.py")

    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()

    old_test_setup = """    model.sampler_config.number_of_uniform_points = 12000
    model.sampler_config.number_of_interface_points = 3000
    model.sampler_config.distance_threshold = 0.005"""

    new_test_setup = """    model.sampler_config.number_of_uniform_points = 8000
    model.sampler_config.number_of_material_points = 7000
    model.sampler_config.distance_threshold = 0.005"""

    if "number_of_interface_points" in test_content:
        test_content = test_content.replace(old_test_setup, new_test_setup)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"[*] Đã cập nhật test_simulation.py")

if __name__ == "__main__":
    update_point_sampling_strategy()