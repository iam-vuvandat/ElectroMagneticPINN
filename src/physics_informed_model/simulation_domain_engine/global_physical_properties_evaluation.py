import torch

VACUUM_RELUCTIVITY = 795774.715459

def evaluate_global_physical_properties(domains_list, points_tensor):
    number_of_points = points_tensor.shape[0]
    computation_device = points_tensor.device

    background_domain = None
    material_domains = []
    for domain in domains_list:
        if getattr(domain, 'domain_name', '') == 'background':
            background_domain = domain
        else:
            material_domains.append(domain)

    if background_domain is not None:
        base_reluctivity = background_domain.evaluate_reluctivity(points_tensor)
        base_hx, base_hy = background_domain.evaluate_magnetization_vector(points_tensor)
        base_jz = background_domain.evaluate_current_density(points_tensor)
        base_pde_weight = background_domain.training_config.pde_weight
        base_interface_weight = background_domain.training_config.interface_weight
        base_sampling_density = background_domain.training_config.sampling_density_multiplier
        base_mat_id = float(background_domain.domain_id) if background_domain.domain_id is not None else float(len(domains_list))
    else:
        base_reluctivity = torch.full((number_of_points, 1), VACUUM_RELUCTIVITY, dtype=torch.float32, device=computation_device)
        base_hx = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
        base_hy = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
        base_jz = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
        base_pde_weight = 1.0
        base_interface_weight = 1.0
        base_sampling_density = 1.0
        base_mat_id = 0.0

    global_reluctivity_tensor = base_reluctivity.clone()
    global_coercive_field_x_tensor = base_hx.clone()
    global_coercive_field_y_tensor = base_hy.clone()
    global_current_density_z_tensor = base_jz.clone()
    global_material_classification_tensor = torch.full((number_of_points, 1), base_mat_id, dtype=torch.float32, device=computation_device)

    global_pde_weight_tensor = torch.full((number_of_points, 1), base_pde_weight, dtype=torch.float32, device=computation_device)
    global_interface_weight_tensor = torch.full((number_of_points, 1), base_interface_weight, dtype=torch.float32, device=computation_device)
    global_sampling_density_tensor = torch.full((number_of_points, 1), base_sampling_density, dtype=torch.float32, device=computation_device)

    if not material_domains:
        return {
            "reluctivity": global_reluctivity_tensor,
            "coercive_field_x": global_coercive_field_x_tensor,
            "coercive_field_y": global_coercive_field_y_tensor,
            "current_density_z": global_current_density_z_tensor,
            "material_classification": global_material_classification_tensor,
            "pde_weight": global_pde_weight_tensor,
            "interface_weight": global_interface_weight_tensor,
            "sampling_density_multiplier": global_sampling_density_tensor
        }

    masks = []
    reluctivities = []
    hxs, hys, jzs = [], [], []

    for domain_object in material_domains:
        sdf = domain_object.compute_signed_distance_field(points_tensor)
        mask = (sdf <= 0.0).float().view(-1, 1)
        masks.append(mask)
        reluctivities.append(domain_object.evaluate_reluctivity(points_tensor))
        hx, hy = domain_object.evaluate_magnetization_vector(points_tensor)
        hxs.append(hx)
        hys.append(hy)
        jzs.append(domain_object.evaluate_current_density(points_tensor))

    total_mask = sum(masks)
    global_envelope_mask = (total_mask > 0.0).float()
    safe_total_mask = total_mask + 1e-12

    blended_reluctivity = torch.zeros_like(global_reluctivity_tensor)
    blended_hx = torch.zeros_like(global_coercive_field_x_tensor)
    blended_hy = torch.zeros_like(global_coercive_field_y_tensor)
    blended_jz = torch.zeros_like(global_current_density_z_tensor)
    blended_mat = torch.zeros_like(global_material_classification_tensor)

    blended_pde_weight = torch.zeros_like(global_pde_weight_tensor)
    blended_interface_weight = torch.zeros_like(global_interface_weight_tensor)
    blended_sampling_density = torch.zeros_like(global_sampling_density_tensor)

    target_h_mag = torch.zeros_like(global_coercive_field_x_tensor)

    for i in range(len(material_domains)):
        weight = masks[i] / safe_total_mask
        blended_reluctivity += weight * reluctivities[i]
        blended_hx += weight * hxs[i]
        blended_hy += weight * hys[i]
        blended_jz += weight * jzs[i]
        domain_id_val = float(material_domains[i].domain_id) if material_domains[i].domain_id is not None else float(i + 1)
        blended_mat += weight * domain_id_val

        blended_pde_weight += weight * material_domains[i].training_config.pde_weight
        blended_interface_weight += weight * material_domains[i].training_config.interface_weight
        blended_sampling_density += weight * material_domains[i].training_config.sampling_density_multiplier

        seg_h_mag = torch.sqrt(hxs[i]**2 + hys[i]**2)
        target_h_mag += weight * seg_h_mag

    blended_h_mag = torch.sqrt(blended_hx**2 + blended_hy**2)
    safe_blended_h_mag = torch.where(blended_h_mag < 1e-6, torch.full_like(blended_h_mag, 1.0), blended_h_mag)
    scale_factor = target_h_mag / safe_blended_h_mag

    has_field_mask = (target_h_mag > 1.0).float()
    blended_hx = blended_hx * (1.0 - has_field_mask) + (blended_hx * scale_factor) * has_field_mask
    blended_hy = blended_hy * (1.0 - has_field_mask) + (blended_hy * scale_factor) * has_field_mask

    global_reluctivity_tensor = (1.0 - global_envelope_mask) * base_reluctivity + global_envelope_mask * blended_reluctivity
    global_coercive_field_x_tensor = (1.0 - global_envelope_mask) * base_hx + global_envelope_mask * blended_hx
    global_coercive_field_y_tensor = (1.0 - global_envelope_mask) * base_hy + global_envelope_mask * blended_hy
    global_current_density_z_tensor = (1.0 - global_envelope_mask) * base_jz + global_envelope_mask * blended_jz
    global_material_classification_tensor = (1.0 - global_envelope_mask) * torch.full_like(global_material_classification_tensor, base_mat_id) + global_envelope_mask * blended_mat

    global_pde_weight_tensor = (1.0 - global_envelope_mask) * base_pde_weight + global_envelope_mask * blended_pde_weight
    global_interface_weight_tensor = (1.0 - global_envelope_mask) * base_interface_weight + global_envelope_mask * blended_interface_weight
    global_sampling_density_tensor = (1.0 - global_envelope_mask) * base_sampling_density + global_envelope_mask * blended_sampling_density

    return {
        "reluctivity": global_reluctivity_tensor,
        "coercive_field_x": global_coercive_field_x_tensor,
        "coercive_field_y": global_coercive_field_y_tensor,
        "current_density_z": global_current_density_z_tensor,
        "material_classification": global_material_classification_tensor,
        "pde_weight": global_pde_weight_tensor,
        "interface_weight": global_interface_weight_tensor,
        "sampling_density_multiplier": global_sampling_density_tensor
    }
