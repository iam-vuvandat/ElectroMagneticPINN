import os

def fix_geometry_import_errors():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    geometry_file = os.path.join(project_root, 'geometry_engine', 'geometry.py')
    
    corrected_geometry_content = """import torch
from geometry_engine.segment.segment import Segment
from geometry_engine.global_signed_distance_field import compute_global_signed_distance_field
from geometry_engine.global_physical_properties_evaluation import evaluate_global_physical_properties
from geometry_engine.geometry_visualizer import display_geometry_problem

class Geometry:
    def __init__(self):
        self.segments_list = []

    def add_segment(self, segment_object):
        segment_object.segment_id = len(self.segments_list) + 1
        self.segments_list.append(segment_object)
        return self

    def get_segment_count(self):
        return len(self.segments_list)

    def compute_global_signed_distance_field(self, points_tensor):
        return compute_global_signed_distance_field(self.segments_list, points_tensor)

    def evaluate_global_physical_properties(self, points_tensor):
        return evaluate_global_physical_properties(self.segments_list, points_tensor)

    def plot_problem_definition(self, x_boundaries_tuple, y_boundaries_tuple, resolution=100):
        display_geometry_problem(self, x_boundaries_tuple, y_boundaries_tuple, resolution)

    def display(self, x_boundaries_tuple, y_boundaries_tuple, resolution=150):
        display_geometry_problem(self, x_boundaries_tuple, y_boundaries_tuple, resolution)
"""

    with open(geometry_file, 'w', encoding='utf-8') as f:
        f.write(corrected_geometry_content)
    print(f"[*] Đã khắc phục lỗi import tại: {geometry_file}")

if __name__ == "__main__":
    fix_geometry_import_errors()