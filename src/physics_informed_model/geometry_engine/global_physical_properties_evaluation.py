import torch

VACUUM_RELUCTIVITY = 795774.715459

def evaluate_global_physical_properties(segments_list, points_tensor):
    number_of_points = points_tensor.shape[0]
    computation_device = points_tensor.device
    
    global_reluctivity_tensor = torch.full((number_of_points, 1), VACUUM_RELUCTIVITY, dtype=torch.float32, device=computation_device)
    global_coercive_field_x_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_coercive_field_y_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_current_density_z_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_material_classification_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)

    if not segments_list:
        return {
            "reluctivity": global_reluctivity_tensor,
            "coercive_field_x": global_coercive_field_x_tensor,
            "coercive_field_y": global_coercive_field_y_tensor,
            "current_density_z": global_current_density_z_tensor,
            "material_classification": global_material_classification_tensor
        }

    sdfs = []
    masks = []
    reluctivities = []
    hxs, hys, jzs = [], [], []

    for segment_object in segments_list:
        sdf = segment_object.compute_signed_distance_field(points_tensor)
        sdfs.append(sdf)
        
        actual_k = 5000.0 - segment_object.steepness * 4960.0
        mask = torch.sigmoid(-actual_k * sdf).view(-1, 1)
        
        masks.append(mask)
        reluctivities.append(segment_object.evaluate_reluctivity(points_tensor))
        hx, hy = segment_object.evaluate_magnetization_vector(points_tensor)
        hxs.append(hx)
        hys.append(hy)
        jzs.append(segment_object.evaluate_current_density(points_tensor))

    stacked_sdfs = torch.stack(sdfs, dim=1).squeeze(-1)
    global_sdf, _ = torch.min(stacked_sdfs, dim=1, keepdim=True)
    
    max_steepness = max([seg.steepness for seg in segments_list])
    envelope_k = 5000.0 - max_steepness * 4960.0
    global_envelope_mask = torch.sigmoid(-envelope_k * global_sdf).view(-1, 1)

    stacked_masks = torch.stack(masks, dim=1)
    stacked_reluctivities = torch.stack(reluctivities, dim=1)
    stacked_hxs = torch.stack(hxs, dim=1)
    stacked_hys = torch.stack(hys, dim=1)
    stacked_jzs = torch.stack(jzs, dim=1)

    total_mask = torch.sum(stacked_masks, dim=1)
    safe_total_mask = total_mask + 1e-12 
    weights = stacked_masks / safe_total_mask.unsqueeze(1) 

    blended_reluctivity = torch.sum(weights * stacked_reluctivities, dim=1)
    blended_hx = torch.sum(weights * stacked_hxs, dim=1)
    blended_hy = torch.sum(weights * stacked_hys, dim=1)
    blended_jz = torch.sum(weights * stacked_jzs, dim=1)
    
    mat_indices = torch.arange(1.0, len(segments_list) + 1.0, device=computation_device).view(1, -1, 1)
    blended_mat = torch.sum(weights * mat_indices, dim=1)

    stacked_h_mag = torch.sqrt(stacked_hxs**2 + stacked_hys**2)
    target_h_mag = torch.sum(weights * stacked_h_mag, dim=1)

    blended_h_mag = torch.sqrt(blended_hx**2 + blended_hy**2)
    safe_blended_h_mag = torch.where(blended_h_mag < 1e-6, torch.full_like(blended_h_mag, 1.0), blended_h_mag)
    scale_factor = target_h_mag / safe_blended_h_mag
    
    has_field_mask = (target_h_mag > 1.0).float()
    blended_hx = blended_hx * (1.0 - has_field_mask) + (blended_hx * scale_factor) * has_field_mask
    blended_hy = blended_hy * (1.0 - has_field_mask) + (blended_hy * scale_factor) * has_field_mask

    global_reluctivity_tensor = VACUUM_RELUCTIVITY + global_envelope_mask * (blended_reluctivity - VACUUM_RELUCTIVITY)
    global_coercive_field_x_tensor = global_envelope_mask * blended_hx
    global_coercive_field_y_tensor = global_envelope_mask * blended_hy
    global_current_density_z_tensor = global_envelope_mask * blended_jz
    global_material_classification_tensor = global_envelope_mask * blended_mat

    return {
        "reluctivity": global_reluctivity_tensor,
        "coercive_field_x": global_coercive_field_x_tensor,
        "coercive_field_y": global_coercive_field_y_tensor,
        "current_density_z": global_current_density_z_tensor,
        "material_classification": global_material_classification_tensor
    }
