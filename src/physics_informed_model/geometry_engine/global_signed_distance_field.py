import torch

def compute_global_signed_distance_field(segments_list, points_tensor):
    if not segments_list:
        return torch.ones((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
        
    global_signed_distance_field = segments_list[0].compute_signed_distance_field(points_tensor)
    
    for segment_object in segments_list[1:]:
        current_signed_distance_field = segment_object.compute_signed_distance_field(points_tensor)
        global_signed_distance_field = torch.minimum(global_signed_distance_field, current_signed_distance_field)
        
    return global_signed_distance_field
