import torch

# Đưa hằng số vật lý ra làm hằng số module (Module-level constant)
VACUUM_RELUCTIVITY = 795774.715459

def evaluate_global_physical_properties(segments_list, points_tensor):
    
    number_of_points = points_tensor.shape[0]
    computation_device = points_tensor.device
    
    global_reluctivity_tensor = torch.full((number_of_points, 1), VACUUM_RELUCTIVITY, dtype=torch.float32, device=computation_device)
    global_coercive_field_x_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_coercive_field_y_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_current_density_z_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_material_classification_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)

    material_index_counter = 1.0

    for segment_object in segments_list:
        signed_distance_field = segment_object.compute_signed_distance_field(points_tensor)
        
        
        actual_k = 5000.0 - segment_object.steepness * 4960.0
        mask_smooth = torch.sigmoid(-actual_k * signed_distance_field).view(-1, 1)
        
        seg_reluctivity = segment_object.evaluate_reluctivity(points_tensor)
        hx_tensor, hy_tensor = segment_object.evaluate_magnetization_vector(points_tensor)
        seg_jz = segment_object.evaluate_current_density(points_tensor)
        
        
        global_reluctivity_tensor = global_reluctivity_tensor + mask_smooth * (seg_reluctivity - VACUUM_RELUCTIVITY)
        global_coercive_field_x_tensor = global_coercive_field_x_tensor + mask_smooth * hx_tensor
        global_coercive_field_y_tensor = global_coercive_field_y_tensor + mask_smooth * hy_tensor
        global_current_density_z_tensor = global_current_density_z_tensor + mask_smooth * seg_jz
        
        global_material_classification_tensor = global_material_classification_tensor + mask_smooth * material_index_counter
        
        material_index_counter += 1.0

    return {
        "reluctivity": global_reluctivity_tensor,
        "coercive_field_x": global_coercive_field_x_tensor,
        "coercive_field_y": global_coercive_field_y_tensor,
        "current_density_z": global_current_density_z_tensor,
        "material_classification": global_material_classification_tensor
    }