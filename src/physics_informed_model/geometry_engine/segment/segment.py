import torch
from geometry_engine.segment.polygon_signed_distance_field import compute_polygon_signed_distance_field

class Segment:
    def __init__(self, vertices_list=None):
        self.material_name = "air"
        self.vacuum_reluctivity = 795774.715459
        
        self.relative_permeability = 1.0
        self.reluctivity_function = None
        
        self.coercive_field_x = 0.0
        self.coercive_field_y = 0.0
        self.magnetization_vector_function = None
        
        self.current_density_z_axis = 0.0
        self.current_density_function = None
        
        self.vertices_tensor = None
        if vertices_list is not None:
            self.set_vertices(vertices_list)

    def set_vertices(self, vertices_list):
        self.vertices_tensor = torch.tensor(vertices_list, dtype=torch.float32)
        return self

    def set_material_properties(
        self, 
        name="default", 
        relative_permeability=1.0, 
        reluctivity_function=None,
        coercive_field_x=0.0,
        coercive_field_y=0.0,
        magnetization_vector_function=None, 
        current_density_z_axis=0.0,
        current_density_function=None
    ):
        self.material_name = name
        self.relative_permeability = relative_permeability
        self.reluctivity_function = reluctivity_function
        self.coercive_field_x = coercive_field_x
        self.coercive_field_y = coercive_field_y
        self.magnetization_vector_function = magnetization_vector_function
        self.current_density_z_axis = current_density_z_axis
        self.current_density_function = current_density_function
        return self

    def compute_signed_distance_field(self, points_tensor):
        return compute_polygon_signed_distance_field(self.vertices_tensor, points_tensor)

    def evaluate_reluctivity(self, points_tensor):
        if self.reluctivity_function is not None:
            return self.reluctivity_function(points_tensor)
        constant_reluctivity = self.vacuum_reluctivity / self.relative_permeability
        return torch.full((points_tensor.shape[0], 1), constant_reluctivity, dtype=torch.float32, device=points_tensor.device)

    def evaluate_magnetization_vector(self, points_tensor):
        if self.magnetization_vector_function is not None:
            return self.magnetization_vector_function(points_tensor)
        hx_tensor = torch.full((points_tensor.shape[0], 1), self.coercive_field_x, dtype=torch.float32, device=points_tensor.device)
        hy_tensor = torch.full((points_tensor.shape[0], 1), self.coercive_field_y, dtype=torch.float32, device=points_tensor.device)
        return hx_tensor, hy_tensor

    def evaluate_current_density(self, points_tensor):
        if self.current_density_function is not None:
            return self.current_density_function(points_tensor)
        return torch.full((points_tensor.shape[0], 1), self.current_density_z_axis, dtype=torch.float32, device=points_tensor.device)
