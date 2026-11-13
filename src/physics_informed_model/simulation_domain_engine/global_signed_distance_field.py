import torch

def compute_global_signed_distance_field(domains_list, points_tensor):
    material_domains = [d for d in domains_list if getattr(d, 'domain_name', '') != 'background']
    if not material_domains:
        return torch.full((points_tensor.shape[0], 1), 1e6, dtype=torch.float32, device=points_tensor.device)
        
    global_signed_distance_field = material_domains[0].compute_signed_distance_field(points_tensor)
    for domain_object in material_domains[1:]:
        current_signed_distance_field = domain_object.compute_signed_distance_field(points_tensor)
        global_signed_distance_field = torch.minimum(global_signed_distance_field, current_signed_distance_field)
        
    return global_signed_distance_field
