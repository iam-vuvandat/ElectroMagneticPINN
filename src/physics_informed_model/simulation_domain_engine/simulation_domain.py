import torch
from simulation_domain_engine.boundary.boundary import Boundary
from simulation_domain_engine.material_domain.material_domain import MaterialDomain
from simulation_domain_engine.global_signed_distance_field import compute_global_signed_distance_field
from simulation_domain_engine.global_physical_properties_evaluation import evaluate_global_physical_properties
from simulation_domain_engine.domain_visualizer import display_simulation_domain

class SimulationDomain:
    def __init__(self, x_boundaries_tuple=(-0.05, 0.05), y_boundaries_tuple=(-0.05, 0.05)):
        self.boundary = Boundary()
        self.boundary.geometry_config.x_boundaries_tuple = x_boundaries_tuple
        self.boundary.geometry_config.y_boundaries_tuple = y_boundaries_tuple
        self.boundary.build_boundary()

        self.domains_list = []
        self.background_domain = None

    @property
    def x_boundaries_tuple(self):
        return self.boundary.geometry_config.x_boundaries_tuple

    @x_boundaries_tuple.setter
    def x_boundaries_tuple(self, bounds):
        self.boundary.geometry_config.x_boundaries_tuple = bounds
        self.boundary.build_boundary()

    @property
    def y_boundaries_tuple(self):
        return self.boundary.geometry_config.y_boundaries_tuple

    @y_boundaries_tuple.setter
    def y_boundaries_tuple(self, bounds):
        self.boundary.geometry_config.y_boundaries_tuple = bounds
        self.boundary.build_boundary()

    def set_boundary(self, boundary_object):
        self.boundary = boundary_object
        return self

    def add_material_domain(self, domain_object):
        if self.background_domain in self.domains_list:
            insert_idx = len(self.domains_list) - 1
            self.domains_list.insert(insert_idx, domain_object)
        else:
            self.domains_list.append(domain_object)
            
        for idx, dom in enumerate(self.domains_list):
            dom.domain_id = idx + 1
        return self

    def build_background_domain(self, domain_name="background", training_config=None, network_config=None):
        if self.background_domain in self.domains_list:
            self.domains_list.remove(self.background_domain)

        x_min, x_max = self.boundary.geometry_config.x_boundaries_tuple
        y_min, y_max = self.boundary.geometry_config.y_boundaries_tuple
        
        box_outline = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]

        bg_domain = MaterialDomain(domain_name=domain_name)
        bg_domain.geometry_config.outline = box_outline
        bg_domain.physics_config.relative_permeability = 1.0
        bg_domain.physics_config.coercive = [0.0, 0.0]
        bg_domain.physics_config.current = 0.0

        if training_config is not None:
            for key, value in vars(training_config).items():
                setattr(bg_domain.training_config, key, value)

        if network_config is not None:
            for key, value in vars(network_config).items():
                setattr(bg_domain.network_config, key, value)

        bg_domain.build_domain()

        total_box_area = (x_max - x_min) * (y_max - y_min)
        total_sub_material_area = sum(d.section_area for d in self.domains_list)
        bg_domain.section_area = max(0.0, total_box_area - total_sub_material_area)

        self.background_domain = bg_domain
        self.domains_list.append(self.background_domain)

        for idx, dom in enumerate(self.domains_list):
            dom.domain_id = idx + 1

        return self.background_domain

    def get_domain_count(self):
        return len(self.domains_list)

    def compute_global_signed_distance_field(self, points_tensor):
        return compute_global_signed_distance_field(self.domains_list, points_tensor)

    def evaluate_global_physical_properties(self, points_tensor):
        return evaluate_global_physical_properties(self.domains_list, points_tensor)

    def plot_problem_definition(self, resolution=100):
        display_simulation_domain(self, self.x_boundaries_tuple, self.y_boundaries_tuple, resolution)

    def display(self, resolution=150):
        display_simulation_domain(self, self.x_boundaries_tuple, self.y_boundaries_tuple, resolution)
