import torch
from geometry_engine.segment.polygon_signed_distance_field import compute_polygon_signed_distance_field

class Segment:
    def __init__(self, outline=None):
        self.outline = None
        self.material = "air"
        self.bh_curve = None
        self.coercive = [0.0, 0.0]
        self.current_density = 0.0
        
        self.vacuum_reluctivity = 795774.715459 # 1/(4*pi* 10^(-7))
        self.relative_permeability = 1.0
        
        self.outline_tensor = None
        if outline is not None:
            self.set_outline(outline)

    def compute_constant_bh_curve(self):
        pass

    def set_outline(self, outline):
        self.outline_tensor = torch.tensor(outline, dtype=torch.float32)
        return self

    def set_material_properties(
        self, 
        material="default", 
        b_h_curve=None,
        coercive=[0.0, 0.0],
        current_density=0.0,
        relative_permeability=1.0
    ):
        self.material = material
        self.b_h_curve = b_h_curve
        self.coercive = coercive
        self.current_density = current_density
        self.relative_permeability = relative_permeability
        return self

    def compute_signed_distance_field(self, points_tensor):
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(self, points_tensor):
        if self.b_h_curve is not None:
            return self.b_h_curve(points_tensor)
        
        constant_reluctivity = self.vacuum_reluctivity / self.relative_permeability
        return torch.full((points_tensor.shape[0], 1), constant_reluctivity, dtype=torch.float32, device=points_tensor.device)

    def evaluate_magnetization_vector(self, points_tensor):
        hx_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[0], dtype=torch.float32, device=points_tensor.device)
        hy_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[1], dtype=torch.float32, device=points_tensor.device)
        return hx_tensor, hy_tensor

    def evaluate_current_density(self, points_tensor):
        return torch.full((points_tensor.shape[0], 1), self.current_density, dtype=torch.float32, device=points_tensor.device)
