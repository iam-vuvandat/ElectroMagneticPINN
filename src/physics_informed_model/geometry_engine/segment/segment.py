import torch
from geometry_engine.segment.polygon_signed_distance_field import compute_polygon_signed_distance_field

class Segment:
    def __init__(self, outline=None):
        self.outline = None
        self.material = "air"
        self.bh_curve = None
        self.coercive = [0.0, 0.0]
        self.current = 0.0
        self.current_density = 0.0
        self.section_area = 0.0
        
        self.vacuum_reluctivity = 795774.715459
        self.relative_permeability = 1.0
        
        self.outline_tensor = None
        if outline is not None:
            self.set_outline(outline)

    def compute_constant_bh_curve(self):
        constant_reluctivity = self.vacuum_reluctivity / self.relative_permeability
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

    def set_outline(self, outline):
        self.outline = outline
        self.outline_tensor = torch.tensor(outline, dtype=torch.float32)
        self.compute_section_area()
        return self

    def compute_signed_distance_field(self, points_tensor):
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(self, points_tensor):
        if getattr(self, 'bh_curve', None) is not None:
            return self.bh_curve(points_tensor)
        
        constant_reluctivity = self.vacuum_reluctivity / self.relative_permeability
        return torch.full((points_tensor.shape[0], 1), constant_reluctivity, dtype=torch.float32, device=points_tensor.device)

    def evaluate_magnetization_vector(self, points_tensor):
        hx_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[0], dtype=torch.float32, device=points_tensor.device)
        hy_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[1], dtype=torch.float32, device=points_tensor.device)
        return hx_tensor, hy_tensor

    def evaluate_current_density(self, points_tensor):
        return torch.full((points_tensor.shape[0], 1), self.current_density, dtype=torch.float32, device=points_tensor.device)