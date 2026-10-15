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

    def _build_split_subsegments(self, domain_object):
        outline = np.array(domain_object.geometry_config.outline, dtype=np.float64)
        edge_starts = outline
        edge_ends = np.roll(edge_starts, shift=-1, axis=0)

        other_vertices = []
        if self.electro_magnetic_pinn is not None:
            sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
            for d in sim_domain.domains_list:
                if d is not domain_object and getattr(d, 'domain_name', '') != 'background':
                    if d.geometry_config.outline is not None:
                        other_vertices.append(np.array(d.geometry_config.outline, dtype=np.float64))
        all_other_verts = np.vstack(other_vertices) if other_vertices else np.empty((0, 2), dtype=np.float64)

        sub_starts = []
        sub_ends = []
        for s_pt, e_pt in zip(edge_starts, edge_ends):
            vec = e_pt - s_pt
            seg_len_sq = float(np.dot(vec, vec))
            if seg_len_sq <= 1e-16:
                continue
            t_splits = [0.0, 1.0]
            if all_other_verts.shape[0] > 0:
                rel = all_other_verts - s_pt
                t_proj = np.dot(rel, vec) / seg_len_sq
                valid_t = (t_proj > 1e-3) & (t_proj < 1.0 - 1e-3)
                if np.any(valid_t):
                    cand_t = t_proj[valid_t]
                    cand_verts = all_other_verts[valid_t]
                    proj_pts = s_pt + cand_t[:, None] * vec[None, :]
                    dists = np.linalg.norm(cand_verts - proj_pts, axis=1)
                    on_edge = dists <= 1e-5
                    if np.any(on_edge):
                        t_splits.extend(cand_t[on_edge].tolist())
            t_unique = np.unique(np.round(np.array(t_splits, dtype=np.float64), decimals=6))
            t_unique.sort()
            for idx in range(len(t_unique) - 1):
                t0, t1 = t_unique[idx], t_unique[idx + 1]
                if t1 - t0 > 1e-4:
                    sub_starts.append(s_pt + t0 * vec)
                    sub_ends.append(s_pt + t1 * vec)

        return np.array(sub_starts, dtype=np.float32), np.array(sub_ends, dtype=np.float32)

    def sample_domain_surface_points(self, domain_object, number_of_points):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        outline = domain_object.geometry_config.outline
        if outline is None or len(outline) < 3:
            raise ValueError(f"Lỗi lấy mẫu bề mặt: Domain '{domain_object.domain_name}' có outline không hợp lệ.")

        sub_starts, sub_ends = self._build_split_subsegments(domain_object)
        if sub_starts.shape[0] == 0:
            raise ValueError(f"Lỗi lấy mẫu bề mặt: Chu vi của domain '{domain_object.domain_name}' bằng 0.")

        sub_vecs = sub_ends - sub_starts
        sub_lens = np.sqrt(np.sum(sub_vecs ** 2, axis=1))
        total_perimeter = float(np.sum(sub_lens))
        if total_perimeter <= 1e-12:
            raise ValueError(f"Lỗi lấy mẫu bề mặt: Chu vi của domain '{domain_object.domain_name}' bằng 0.")

        num_subs = sub_starts.shape[0]
        raw_counts = (sub_lens / total_perimeter) * number_of_points
        counts = np.floor(raw_counts).astype(int)
        remainder = int(number_of_points - np.sum(counts))
        if remainder > 0:
            fracs = raw_counts - counts
            top_indices = np.argsort(-fracs)[:remainder]
            counts[top_indices] += 1

        pts_list = []
        for idx in range(num_subs):
            c = int(counts[idx])
            if c <= 0:
                continue
            u_base = (np.arange(c, dtype=np.float32) + 0.5) / float(c)
            t_local = (0.02 + 0.96 * u_base)[:, None]
            seg_pts = sub_starts[idx:idx+1, :] + t_local * sub_vecs[idx:idx+1, :]
            pts_list.append(seg_pts)

        pts_sampled = np.vstack(pts_list) if pts_list else np.empty((0, 2), dtype=np.float32)
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

        tight_pad_x = 0.03 * (x_max - x_min)
        tight_pad_y = 0.03 * (y_max - y_min)
        tx_min = max(x_min, mx_min - tight_pad_x)
        tx_max = min(x_max, mx_max + tight_pad_x)
        ty_min = max(y_min, my_min - tight_pad_y)
        ty_max = min(y_max, my_max + tight_pad_y)

        n_tight = int(round(number_of_points * 0.25))
        n_focus = int(round(number_of_points * 0.35))
        n_uniform = number_of_points - n_focus - n_tight

        pts_uniform = self._sample_air_box(simulation_domain_object, n_uniform, x_min, x_max, y_min, y_max)
        pts_focus = self._sample_air_box(simulation_domain_object, n_focus, gx_min, gx_max, gy_min, gy_max)
        pts_tight = self._sample_air_box(simulation_domain_object, n_tight, tx_min, tx_max, ty_min, ty_max)
        return torch.cat([pts_uniform, pts_focus, pts_tight], dim=0)

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
