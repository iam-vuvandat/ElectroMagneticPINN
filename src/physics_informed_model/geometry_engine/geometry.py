import torch
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
