import torch
import numpy as np

class CollocationSampler:
    def __init__(self, electro_magnetic_pinn=None, device=None):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        if device is not None:
            self.device = device
        elif electro_magnetic_pinn is not None and hasattr(electro_magnetic_pinn, 'computation_device'):
            self.device = electro_magnetic_pinn.computation_device
        else:
            self.device = torch.device('cpu')

    def sample_domain_interior_points(self, domain_object, bounding_box, number_of_points):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)
            
        outline = np.array(domain_object.geometry_config.outline)
        x_min, x_max = np.min(outline[:, 0]), np.max(outline[:, 0])
        y_min, y_max = np.min(outline[:, 1]), np.max(outline[:, 1])

        collected_points = []
        collected_count = 0
        pool_size = max(number_of_points * 4, 500)

        while collected_count < number_of_points:
            points_pool = torch.rand((pool_size, 2), dtype=torch.float32, device=self.device)
            points_pool[:, 0] = points_pool[:, 0] * (x_max - x_min) + x_min
            points_pool[:, 1] = points_pool[:, 1] * (y_max - y_min) + y_min
            
            sdf = domain_object.compute_signed_distance_field(points_pool)
            inside_mask = sdf <= 0.0
            
            if inside_mask.any():
                valid = points_pool[inside_mask]
                collected_points.append(valid)
                collected_count += valid.shape[0]

        pts = torch.cat(collected_points, dim=0)[:number_of_points, :]
        return pts

    def sample_domain_surface_points(self, domain_object, number_of_points):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        outline = domain_object.geometry_config.outline
        num_vertices = len(outline)
        
        edge_starts = np.array(outline)
        edge_ends = np.roll(edge_starts, shift=-1, axis=0)
        
        edge_vectors = edge_ends - edge_starts
        edge_lengths = np.sqrt(np.sum(edge_vectors**2, axis=1))
        total_perimeter = np.sum(edge_lengths)
        
        edge_probs = edge_lengths / total_perimeter
        
        sampled_edges = np.random.choice(num_vertices, size=number_of_points, p=edge_probs)
        t_params = np.random.rand(number_of_points, 1)
        
        pts_sampled = edge_starts[sampled_edges] + t_params * edge_vectors[sampled_edges]
        return torch.tensor(pts_sampled, dtype=torch.float32, device=self.device)

    def sample_background_interior_points(self, simulation_domain_object, number_of_points):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        x_min, x_max = simulation_domain_object.x_boundaries_tuple
        y_min, y_max = simulation_domain_object.y_boundaries_tuple

        collected_points = []
        collected_count = 0
        pool_size = max(number_of_points * 2, 1000)

        while collected_count < number_of_points:
            points_pool = torch.rand((pool_size, 2), dtype=torch.float32, device=self.device)
            points_pool[:, 0] = points_pool[:, 0] * (x_max - x_min) + x_min
            points_pool[:, 1] = points_pool[:, 1] * (y_max - y_min) + y_min

            global_sdf = simulation_domain_object.compute_global_signed_distance_field(points_pool)
            air_mask = global_sdf > 0.0

            if air_mask.any():
                valid = points_pool[air_mask]
                collected_points.append(valid)
                collected_count += valid.shape[0]

        pts = torch.cat(collected_points, dim=0)[:number_of_points, :]
        return pts

    def generate_all_domain_points(self, simulation_domain_object=None):
        if simulation_domain_object is None:
            simulation_domain_object = self.electro_magnetic_pinn.simulation_domain_engine_instance

        domains_data = {}
        bb = (simulation_domain_object.x_boundaries_tuple, simulation_domain_object.y_boundaries_tuple)

        for domain in simulation_domain_object.domains_list:
            if getattr(domain, 'domain_name', '') == 'background':
                n_interior = domain.training_config.number_of_interior_points
                interior_pts = self.sample_background_interior_points(simulation_domain_object, n_interior)
                surface_pts = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            else:
                n_interior = domain.training_config.number_of_interior_points
                n_surface = domain.training_config.number_of_surface_points
                interior_pts = self.sample_domain_interior_points(domain, bb, n_interior)
                surface_pts = self.sample_domain_surface_points(domain, n_surface)

            interior_pts = interior_pts.detach().clone().requires_grad_(True)
            surface_pts = surface_pts.detach().clone().requires_grad_(True)

            key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{domain.domain_id}'
            domains_data[key] = {
                "domain_object": domain,
                "interior_points": interior_pts,
                "surface_points": surface_pts
            }

        return domains_data
