def compute_boundary_factor(self, points_tensor, domain_scale=None):
    x_min, x_max = self.geometry_config.x_boundaries_tuple
    y_min, y_max = self.geometry_config.y_boundaries_tuple
    x = points_tensor[:, 0:1]
    y = points_tensor[:, 1:2]
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y_norm = 2.0 * (y - y_min) / (y_max - y_min) - 1.0
    x_factor = 1.0 - x_norm ** 2
    y_factor = 1.0 - y_norm ** 2
    return x_factor * y_factor
