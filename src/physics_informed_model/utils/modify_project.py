import os

def execute_performance_optimization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    eval_file = os.path.join(project_root, 'geometry_engine', 'global_physical_properties_evaluation.py')
    eval_code = """import torch

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

    total_mask = sum(masks)
    safe_total_mask = total_mask + 1e-12 

    blended_reluctivity = torch.zeros_like(global_reluctivity_tensor)
    blended_hx = torch.zeros_like(global_coercive_field_x_tensor)
    blended_hy = torch.zeros_like(global_coercive_field_y_tensor)
    blended_jz = torch.zeros_like(global_current_density_z_tensor)
    blended_mat = torch.zeros_like(global_material_classification_tensor)
    
    target_h_mag = torch.zeros_like(global_coercive_field_x_tensor)

    for i in range(len(segments_list)):
        weight = masks[i] / safe_total_mask
        
        blended_reluctivity += weight * reluctivities[i]
        blended_hx += weight * hxs[i]
        blended_hy += weight * hys[i]
        blended_jz += weight * jzs[i]
        blended_mat += weight * (i + 1.0)
        
        seg_h_mag = torch.sqrt(hxs[i]**2 + hys[i]**2)
        target_h_mag += weight * seg_h_mag

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
"""

    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(eval_code)

if __name__ == '__main__':
    execute_performance_optimization()