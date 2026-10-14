import torch
import numpy as np

class CollocationSampler:
    def __init__(self, electro_magnetic_pinn=None, device=None, seed=42, max_sampling_attempts=200):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        self.seed = seed
        self.max_sampling_attempts = max_sampling_attempts
        self.sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=self.seed)
        if device is not None:
            self.device = device
        elif electro_magnetic_pinn is not None and hasattr(electro_magnetic_pinn, 'computation_device'):
            self.device = electro_magnetic_pinn.computation_device
        else:
            self.device = torch.device('cpu')

    def _draw_sobol_box(self, count, x_min, x_max, y_min, y_max):
        uv = self.sobol_engine.draw(count).to(device=self.device, dtype=torch.float32)
        pts = torch.empty_like(uv)
        pts[:, 0] = uv[:, 0] * (x_max - x_min) + x_min
        pts[:, 1] = uv[:, 1] * (y_max - y_min) + y_min
        return pts

    def sample_domain_interior_points(self, domain_object, bounding_box, number_of_points):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)
            
        if domain_object.geometry_config.outline is None or len(domain_object.geometry_config.outline) < 3:
            raise ValueError(f"Lỗi lấy mẫu: Domain '{domain_object.domain_name}' có outline không hợp lệ.")
        if domain_object.section_area <= 1e-12:
            raise ValueError(f"Lỗi lấy mẫu: Domain '{domain_object.domain_name}' có diện tích bằng 0.")

        outline = np.array(domain_object.geometry_config.outline)
        x_min, x_max = float(np.min(outline[:, 0])), float(np.max(outline[:, 0]))
        y_min, y_max = float(np.min(outline[:, 1])), float(np.max(outline[:, 1]))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Lỗi lấy mẫu: Bounding box của domain '{domain_object.domain_name}' suy biến.")

        collected_points = []
        collected_count = 0
        pool_size = max(number_of_points * 4, 500)
        attempts = 0

        while collected_count < number_of_points:
            if attempts >= self.max_sampling_attempts:
                raise RuntimeError(
                    f"Lỗi lấy mẫu: Vượt quá {self.max_sampling_attempts} lần thử cho domain '{domain_object.domain_name}'."
                )
            attempts += 1
            points_pool = self._draw_sobol_box(pool_size, x_min, x_max, y_min, y_max)
            sdf = domain_object.compute_signed_distance_field(points_pool).view(-1)
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
        if outline is None or len(outline) < 3:
            raise ValueError(f"Lỗi lấy mẫu bề mặt: Domain '{domain_object.domain_name}' có outline không hợp lệ.")

        num_vertices = len(outline)
        edge_starts = np.array(outline, dtype=np.float32)
        edge_ends = np.roll(edge_starts, shift=-1, axis=0)
        
        edge_vectors = edge_ends - edge_starts
        edge_lengths = np.sqrt(np.sum(edge_vectors ** 2, axis=1))
        total_perimeter = float(np.sum(edge_lengths))
        if total_perimeter <= 1e-12:
            raise ValueError(f"Lỗi lấy mẫu bề mặt: Chu vi của domain '{domain_object.domain_name}' bằng 0.")
        
        edge_probs = edge_lengths / total_perimeter
        sampled_edges = np.random.choice(num_vertices, size=number_of_points, p=edge_probs)
        
        uv_1d = self.sobol_engine.draw(number_of_points)[:, 0:1].cpu().numpy()
        t_params = 0.02 + 0.96 * uv_1d
        
        pts_sampled = edge_starts[sampled_edges] + t_params * edge_vectors[sampled_edges]
        return torch.tensor(pts_sampled, dtype=torch.float32, device=self.device)

    def _sample_air_box(self, simulation_domain_object, target_count, x_min, x_max, y_min, y_max):
        if target_count <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)
        collected_points = []
        collected_count = 0
        pool_size = max(target_count * 2, 1000)
        attempts = 0
        while collected_count < target_count:
            if attempts >= self.max_sampling_attempts:
                raise RuntimeError(
                    f"Lỗi lấy mẫu: Vượt quá {self.max_sampling_attempts} lần thử cho miền background."
                )
            attempts += 1
            points_pool = self._draw_sobol_box(pool_size, x_min, x_max, y_min, y_max)
            global_sdf = simulation_domain_object.compute_global_signed_distance_field(points_pool).view(-1)
            air_mask = global_sdf > 0.0
            if air_mask.any():
                valid = points_pool[air_mask]
                collected_points.append(valid)
                collected_count += valid.shape[0]
        return torch.cat(collected_points, dim=0)[:target_count, :]

    def sample_background_interior_points(self, simulation_domain_object, number_of_points):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        x_min, x_max = simulation_domain_object.x_boundaries_tuple
        y_min, y_max = simulation_domain_object.y_boundaries_tuple
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Lỗi lấy mẫu: Khung biên của SimulationDomain không hợp lệ.")

        mat_domains = [d for d in simulation_domain_object.domains_list if getattr(d, 'domain_name', '') != 'background']
        if not mat_domains:
            return self._sample_air_box(simulation_domain_object, number_of_points, x_min, x_max, y_min, y_max)

        all_outlines = np.vstack([np.array(d.geometry_config.outline) for d in mat_domains])
        mx_min, mx_max = float(np.min(all_outlines[:, 0])), float(np.max(all_outlines[:, 0]))
        my_min, my_max = float(np.min(all_outlines[:, 1])), float(np.max(all_outlines[:, 1]))
        pad_x = 0.15 * (x_max - x_min)
        pad_y = 0.15 * (y_max - y_min)
        gx_min = max(x_min, mx_min - pad_x)
        gx_max = min(x_max, mx_max + pad_x)
        gy_min = max(y_min, my_min - pad_y)
        gy_max = min(y_max, my_max + pad_y)

        n_focus = int(round(number_of_points * 0.35))
        n_uniform = number_of_points - n_focus

        pts_uniform = self._sample_air_box(simulation_domain_object, n_uniform, x_min, x_max, y_min, y_max)
        pts_focus = self._sample_air_box(simulation_domain_object, n_focus, gx_min, gx_max, gy_min, gy_max)
        return torch.cat([pts_uniform, pts_focus], dim=0)

    def generate_all_domain_points(self, simulation_domain_object=None):
        if simulation_domain_object is None:
            simulation_domain_object = self.electro_magnetic_pinn.simulation_domain_engine_instance

        domains_data = {}
        bb = (simulation_domain_object.x_boundaries_tuple, simulation_domain_object.y_boundaries_tuple)

        for domain in simulation_domain_object.domains_list:
            mult = float(getattr(domain.training_config, 'sampling_density_multiplier', 1.0))
            if mult < 0.0:
                raise ValueError(f"Lỗi: sampling_density_multiplier của domain '{domain.domain_name}' không được âm.")

            base_interior = int(domain.training_config.number_of_interior_points)
            base_surface = int(domain.training_config.number_of_surface_points)

            actual_interior = int(round(base_interior * mult)) if base_interior > 0 else 0
            actual_surface = int(round(base_surface * mult)) if base_surface > 0 else 0

            if getattr(domain, 'domain_name', '') == 'background':
                interior_pts = self.sample_background_interior_points(simulation_domain_object, actual_interior)
                surface_pts = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            else:
                interior_pts = self.sample_domain_interior_points(domain, bb, actual_interior)
                surface_pts = self.sample_domain_surface_points(domain, actual_surface)

            interior_pts = interior_pts.detach().clone().requires_grad_(True)
            surface_pts = surface_pts.detach().clone().requires_grad_(True)

            key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{domain.domain_id}'
            domains_data[key] = {
                "domain_object": domain,
                "interior_points": interior_pts,
                "surface_points": surface_pts
            }

        return domains_data
