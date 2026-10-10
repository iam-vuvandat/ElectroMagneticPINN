import torch

class CollocationSampler:
    def __init__(self, x_boundaries_tuple, y_boundaries_tuple, device=torch.device('cpu')):
        self.x_minimum = x_boundaries_tuple[0]
        self.x_maximum = x_boundaries_tuple[1]
        self.y_minimum = y_boundaries_tuple[0]
        self.y_maximum = y_boundaries_tuple[1]
        self.device = device # Nhận cấu hình thiết bị từ hệ thống

    def generate_uniform_points_tensor(self, number_of_points):
        points_tensor = torch.rand((number_of_points, 2), dtype=torch.float32, device=self.device)
        points_tensor[:, 0] = points_tensor[:, 0] * (self.x_maximum - self.x_minimum) + self.x_minimum
        points_tensor[:, 1] = points_tensor[:, 1] * (self.y_maximum - self.y_minimum) + self.y_minimum
        points_tensor.requires_grad_(True)
        return points_tensor

    def generate_material_points_tensor(self, geometry_object, number_of_points, distance_threshold):
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
        return combined_points_tensor
