def compute_boundary_factor(points_tensor, domain_scale):
    x = points_tensor[:, 0:1]
    y = points_tensor[:, 1:2]
    x_factor = 1.0 - (x / domain_scale) ** 2
    y_factor = 1.0 - (y / domain_scale) ** 2
    return x_factor * y_factor
