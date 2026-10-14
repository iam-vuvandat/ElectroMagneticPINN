import torch

VACUUM_RELUCTIVITY = 795774.715459

def evaluate_global_physical_properties(domains_list, points_tensor):
    number_of_points = points_tensor.shape[0]
    computation_device = points_tensor.device
    computation_dtype = points_tensor.dtype

    background_domain = None
    material_domains = []
    for domain in domains_list:
        if getattr(domain, 'domain_name', '') == 'background':
            background_domain = domain
        else:
            material_domains.append(domain)

    if background_domain is not None:
        global_reluctivity = background_domain.evaluate_reluctivity(points_tensor).clone()
        base_hx, base_hy = background_domain.evaluate_magnetization_vector(points_tensor)
        global_hx = base_hx.clone()
        global_hy = base_hy.clone()
        global_jz = background_domain.evaluate_current_density(points_tensor).clone()
        base_pde_weight = float(background_domain.training_config.pde_weight)
        base_interface_weight = float(background_domain.training_config.interface_weight)
        base_sampling_density = float(background_domain.training_config.sampling_density_multiplier)
        base_mat_id = float(background_domain.domain_id) if background_domain.domain_id is not None else float(len(domains_list))
    else:
        global_reluctivity = torch.full((number_of_points, 1), VACUUM_RELUCTIVITY, dtype=computation_dtype, device=computation_device)
        global_hx = torch.zeros((number_of_points, 1), dtype=computation_dtype, device=computation_device)
        global_hy = torch.zeros((number_of_points, 1), dtype=computation_dtype, device=computation_device)
        global_jz = torch.zeros((number_of_points, 1), dtype=computation_dtype, device=computation_device)
        base_pde_weight = 1.0
        base_interface_weight = 1.0
        base_sampling_density = 1.0
        base_mat_id = 0.0

    global_mat_id = torch.full((number_of_points, 1), base_mat_id, dtype=computation_dtype, device=computation_device)
    global_pde_weight = torch.full((number_of_points, 1), base_pde_weight, dtype=computation_dtype, device=computation_device)
    global_interface_weight = torch.full((number_of_points, 1), base_interface_weight, dtype=computation_dtype, device=computation_device)
    global_sampling_density = torch.full((number_of_points, 1), base_sampling_density, dtype=computation_dtype, device=computation_device)

    if not material_domains:
        return {
            "reluctivity": global_reluctivity,
            "coercive_field_x": global_hx,
            "coercive_field_y": global_hy,
            "current_density_z": global_jz,
            "material_classification": global_mat_id,
            "pde_weight": global_pde_weight,
            "interface_weight": global_interface_weight,
            "sampling_density_multiplier": global_sampling_density
        }

    best_sdf = torch.full((number_of_points, 1), 1e9, dtype=computation_dtype, device=computation_device)

    for idx, domain_object in enumerate(material_domains):
        sdf = domain_object.compute_signed_distance_field(points_tensor).view(-1, 1)
        is_inside_and_closer = (sdf <= 0.0) & (sdf < best_sdf)
        best_sdf = torch.where(is_inside_and_closer, sdf, best_sdf)

        nu_dom = domain_object.evaluate_reluctivity(points_tensor)
        hx_dom, hy_dom = domain_object.evaluate_magnetization_vector(points_tensor)
        jz_dom = domain_object.evaluate_current_density(points_tensor)
        domain_id_val = float(domain_object.domain_id) if domain_object.domain_id is not None else float(idx + 1)

        global_reluctivity = torch.where(is_inside_and_closer, nu_dom, global_reluctivity)
        global_hx = torch.where(is_inside_and_closer, hx_dom, global_hx)
        global_hy = torch.where(is_inside_and_closer, hy_dom, global_hy)
        global_jz = torch.where(is_inside_and_closer, jz_dom, global_jz)
        global_mat_id = torch.where(is_inside_and_closer, torch.full_like(global_mat_id, domain_id_val), global_mat_id)
        global_pde_weight = torch.where(is_inside_and_closer, torch.full_like(global_pde_weight, float(domain_object.training_config.pde_weight)), global_pde_weight)
        global_interface_weight = torch.where(is_inside_and_closer, torch.full_like(global_interface_weight, float(domain_object.training_config.interface_weight)), global_interface_weight)
        global_sampling_density = torch.where(is_inside_and_closer, torch.full_like(global_sampling_density, float(domain_object.training_config.sampling_density_multiplier)), global_sampling_density)

    return {
        "reluctivity": global_reluctivity,
        "coercive_field_x": global_hx,
        "coercive_field_y": global_hy,
        "current_density_z": global_jz,
        "material_classification": global_mat_id,
        "pde_weight": global_pde_weight,
        "interface_weight": global_interface_weight,
        "sampling_density_multiplier": global_sampling_density
    }
