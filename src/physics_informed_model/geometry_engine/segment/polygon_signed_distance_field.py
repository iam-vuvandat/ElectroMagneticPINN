import torch

def compute_polygon_signed_distance_field(vertices_tensor, points_tensor):
    if vertices_tensor is None:
        raise ValueError("Segment vertices must be set before computing signed distance field.")
        
    computation_device = points_tensor.device
    computation_dtype = points_tensor.dtype

    vertices_on_device = vertices_tensor.to(
        device=computation_device,
        dtype=computation_dtype,
    )
    
    start_points = vertices_on_device
    end_points = torch.roll(vertices_on_device, shifts=-1, dims=0)
    
    edge_vectors = end_points - start_points
    point_to_start_vectors = points_tensor.unsqueeze(1) - start_points.unsqueeze(0)
    
    dot_product_point_to_start_and_edge = torch.sum(point_to_start_vectors * edge_vectors.unsqueeze(0), dim=2)
    dot_product_edge_and_edge = torch.sum(edge_vectors * edge_vectors, dim=1).unsqueeze(0)
    
    dot_product_edge_and_edge = torch.clamp(
        dot_product_edge_and_edge,
        min=torch.finfo(computation_dtype).eps,
    )
    
    projection_parameter = dot_product_point_to_start_and_edge / dot_product_edge_and_edge
    clamped_projection_parameter = torch.clamp(projection_parameter, min=0.0, max=1.0)
    
    closest_points = start_points.unsqueeze(0) + clamped_projection_parameter.unsqueeze(-1) * edge_vectors.unsqueeze(0)
    
    diff_vectors = points_tensor.unsqueeze(1) - closest_points
    distances_to_edges = torch.sqrt(torch.sum(diff_vectors * diff_vectors, dim=2) + 1e-12)
    
    minimum_distances, _ = torch.min(distances_to_edges, dim=1)
    
    points_x_coordinates = points_tensor[:, 0].unsqueeze(1)
    points_y_coordinates = points_tensor[:, 1].unsqueeze(1)
    start_points_x_coordinates = start_points[:, 0].unsqueeze(0)
    start_points_y_coordinates = start_points[:, 1].unsqueeze(0)
    end_points_x_coordinates = end_points[:, 0].unsqueeze(0)
    end_points_y_coordinates = end_points[:, 1].unsqueeze(0)
    
    condition_y_between_start_and_end = (start_points_y_coordinates <= points_y_coordinates) & (points_y_coordinates < end_points_y_coordinates)
    condition_y_between_end_and_start = (end_points_y_coordinates <= points_y_coordinates) & (points_y_coordinates < start_points_y_coordinates)
    valid_y_intersection = condition_y_between_start_and_end | condition_y_between_end_and_start
    
    y_difference = end_points_y_coordinates - start_points_y_coordinates
    
    dy_safe = torch.where(
        torch.abs(y_difference) < torch.finfo(computation_dtype).eps,
        torch.full_like(y_difference, 1e-7),
        y_difference,
    )
    
    intersection_x_coordinates = start_points_x_coordinates + (points_y_coordinates - start_points_y_coordinates) * (end_points_x_coordinates - start_points_x_coordinates) / dy_safe
    ray_crossings = valid_y_intersection & (points_x_coordinates < intersection_x_coordinates)
    
    is_point_inside = ray_crossings.sum(dim=1) % 2 == 1
    distance_sign = torch.where(
        is_point_inside,
        -torch.ones_like(minimum_distances),
        torch.ones_like(minimum_distances),
    )
    
    return distance_sign * minimum_distances
