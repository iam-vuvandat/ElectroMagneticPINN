import os

def execute_fix_tensor_dimension_bug():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # ---------------------------------------------------------
    # 1. CẬP NHẬT FILE: GEOMETRY_ENGINE/GEOMETRY.PY
    # (Loại bỏ squeeze(-1) và thay bằng view(-1) để an toàn)
    # ---------------------------------------------------------
    geometry_file = os.path.join(project_root, 'geometry_engine', 'geometry.py')
    geometry_code = """import torch
from geometry_engine.segment.segment import Segment
from geometry_engine.global_signed_distance_field import compute_global_signed_distance_field
from geometry_engine.global_physical_properties_evaluation import evaluate_global_physical_properties
from geometry_engine.geometry_visualizer import plot_geometry_problem

class Geometry:
    def __init__(self):
        self.segments_list = []

    def add_segment(self, segment_object):
        self.segments_list.append(segment_object)
        return self

    def unite(self, segments_collection):
        united_segment = Segment()
        united_segment.components = []
        
        for item in segments_collection:
            if isinstance(item, Segment):
                united_segment.components.append(item)
            else:
                united_segment.components.append(Segment(outline=item))

        def compute_united_sdf(points_tensor):
            if not united_segment.components:
                return torch.ones((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            res_sdf = united_segment.components[0].compute_signed_distance_field(points_tensor)
            for comp in united_segment.components[1:]:
                res_sdf = torch.minimum(res_sdf, comp.compute_signed_distance_field(points_tensor))
            return res_sdf

        united_segment.compute_signed_distance_field = compute_united_sdf

        def evaluate_united_reluctivity(points_tensor):
            sdfs = [comp.compute_signed_distance_field(points_tensor).view(-1) for comp in united_segment.components]
            stacked_sdfs = torch.stack(sdfs, dim=1)
            weights = torch.softmax(-100.0 * stacked_sdfs, dim=1)
            
            blended_nu = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            for i, comp in enumerate(united_segment.components):
                w_i = weights[:, i:i+1]
                blended_nu += w_i * comp.evaluate_reluctivity(points_tensor)
            return blended_nu

        united_segment.evaluate_reluctivity = evaluate_united_reluctivity

        def evaluate_united_magnetization(points_tensor):
            sdfs = [comp.compute_signed_distance_field(points_tensor).view(-1) for comp in united_segment.components]
            stacked_sdfs = torch.stack(sdfs, dim=1)
            weights = torch.softmax(-100.0 * stacked_sdfs, dim=1)
            
            blended_hx = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            blended_hy = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            for i, comp in enumerate(united_segment.components):
                w_i = weights[:, i:i+1]
                hx, hy = comp.evaluate_magnetization_vector(points_tensor)
                blended_hx += w_i * hx
                blended_hy += w_i * hy
            return blended_hx, blended_hy

        united_segment.evaluate_magnetization_vector = evaluate_united_magnetization

        def evaluate_united_current_density(points_tensor):
            sdfs = [comp.compute_signed_distance_field(points_tensor).view(-1) for comp in united_segment.components]
            stacked_sdfs = torch.stack(sdfs, dim=1)
            weights = torch.softmax(-100.0 * stacked_sdfs, dim=1)
            
            blended_jz = torch.zeros((points_tensor.shape[0], 1), dtype=torch.float32, device=points_tensor.device)
            for i, comp in enumerate(united_segment.components):
                w_i = weights[:, i:i+1]
                blended_jz += w_i * comp.evaluate_current_density(points_tensor)
            return blended_jz

        united_segment.evaluate_current_density = evaluate_united_current_density

        united_segment.steepness = max([comp.steepness for comp in united_segment.components]) if united_segment.components else 1.0

        self.segments_list.append(united_segment)
        return self

    def compute_global_signed_distance_field(self, points_tensor):
        return compute_global_signed_distance_field(self.segments_list, points_tensor)

    def evaluate_global_physical_properties(self, points_tensor):
        return evaluate_global_physical_properties(self.segments_list, points_tensor)

    def plot_problem_definition(self, x_boundaries_tuple, y_boundaries_tuple, resolution=100):
        plot_geometry_problem(self, x_boundaries_tuple, y_boundaries_tuple, resolution)
"""

    # ---------------------------------------------------------
    # 2. CẬP NHẬT FILE: GEOMETRY_ENGINE/GLOBAL_PHYSICAL_PROPERTIES_EVALUATION.PY
    # (Loại bỏ squeeze(-1) ở phần tạo stacked_sdfs)
    # ---------------------------------------------------------
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

    # Đảm bảo mỗi SDF đều là 1D trước khi xếp chồng, tạo ra shape (N, M) an toàn tuyệt đối
    stacked_sdfs = torch.stack([s.view(-1) for s in sdfs], dim=1)
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

    with open(geometry_file, 'w', encoding='utf-8') as f:
        f.write(geometry_code)
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(eval_code)

if __name__ == '__main__':
    execute_fix_tensor_dimension_bug()