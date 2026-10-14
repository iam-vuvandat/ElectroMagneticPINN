import torch

def build_boundary(self):
    x_min, x_max = self.geometry_config.x_boundaries_tuple
    y_min, y_max = self.geometry_config.y_boundaries_tuple
    self.bounding_box = (x_min, x_max, y_min, y_max)
    
    box_outline = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]
    self.outline_tensor = torch.tensor(box_outline, dtype=torch.float32)
    return self
