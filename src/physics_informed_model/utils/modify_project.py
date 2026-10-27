import os

def apply_shared_interface_prepartition_updates():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    physics_dir = os.path.join(project_root, 'physics_domain')
    equations_dir = os.path.join(physics_dir, 'physical_equations')
    training_dir = os.path.join(project_root, 'training_engine')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: physics_domain/collocation_sampler.py
    # (Phân loại trước Outer Boundary Points và Shared Interface Points)
    # =========================================================================
    files_to_update[os.path.join(physics_dir, 'collocation_sampler.py')] = '''import torch
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

    def sample_subsegments_points(self, sub_starts, sub_ends, number_of_points):
        if number_of_points <= 0 or sub_starts.shape[0] == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        sub_vecs = sub_ends - sub_starts
        sub_lens = np.sqrt(np.sum(sub_vecs ** 2, axis=1))
        total_perimeter = float(np.sum(sub_lens))
        if total_perimeter <= 1e-12:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

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

    def sample_domain_outer_surface_points(self, domain_object, number_of_points):
        sub_starts, sub_ends = self._build_split_subsegments(domain_object)
        if sub_starts.shape[0] == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        mids = 0.5 * (sub_starts + sub_ends)
        mids_tensor = torch.tensor(mids, dtype=torch.float32, device=self.device)
        
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        other_domains = [
            d for d in sim_domain.domains_list
            if d is not domain_object and getattr(d, 'domain_name', '') != 'background'
        ]

        is_outer = np.ones(sub_starts.shape[0], dtype=bool)
        if other_domains and mids_tensor.shape[0] > 0:
            for other in other_domains:
                sdf = other.compute_signed_distance_field(mids_tensor).view(-1).cpu().numpy()
                is_outer = is_outer & (np.abs(sdf) > 1e-4)

        outer_starts = sub_starts[is_outer]
        outer_ends = sub_ends[is_outer]
        return self.sample_subsegments_points(outer_starts, outer_ends, number_of_points)

    def sample_shared_interface_points(self, domain_a, domain_b, number_of_points):
        sub_starts, sub_ends = self._build_split_subsegments(domain_a)
        if sub_starts.shape[0] == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        mids = 0.5 * (sub_starts + sub_ends)
        mids_tensor = torch.tensor(mids, dtype=torch.float32, device=self.device)
        sdf_b = domain_b.compute_signed_distance_field(mids_tensor).view(-1).cpu().numpy()

        is_shared = np.abs(sdf_b) <= 1e-4
        shared_starts = sub_starts[is_shared]
        shared_ends = sub_ends[is_shared]
        return self.sample_subsegments_points(shared_starts, shared_ends, number_of_points)

    def sample_domain_surface_points(self, domain_object, number_of_points):
        # Tương thích ngược
        sub_starts, sub_ends = self._build_split_subsegments(domain_object)
        return self.sample_subsegments_points(sub_starts, sub_ends, number_of_points)

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

        domains_data = {
            "domains": {},
            "shared_interfaces": {}
        }
        bb = (simulation_domain_object.x_boundaries_tuple, simulation_domain_object.y_boundaries_tuple)
        mat_domains = [d for d in simulation_domain_object.domains_list if getattr(d, 'domain_name', '') != 'background']

        for domain in simulation_domain_object.domains_list:
            mult = float(getattr(domain.training_config, 'sampling_density_multiplier', 1.0))
            base_interior = int(domain.training_config.number_of_interior_points)
            base_surface = int(domain.training_config.number_of_surface_points)

            actual_interior = int(round(base_interior * mult)) if base_interior > 0 else 0
            actual_surface = int(round(base_surface * mult)) if base_surface > 0 else 0

            if getattr(domain, 'domain_name', '') == 'background':
                interior_pts = self.sample_background_interior_points(simulation_domain_object, actual_interior)
                outer_pts = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            else:
                interior_pts = self.sample_domain_interior_points(domain, bb, actual_interior)
                outer_pts = self.sample_domain_outer_surface_points(domain, actual_surface)

            key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{domain.domain_id}'
            domains_data["domains"][key] = {
                "domain_object": domain,
                "interior_points": interior_pts.detach().clone().requires_grad_(True),
                "outer_surface_points": outer_pts.detach().clone().requires_grad_(True),
                "surface_points": outer_pts.detach().clone().requires_grad_(True)
            }

        # Tạo điểm giao diện chia trước cho các cặp vật liệu giáp kề nhau
        for i in range(len(mat_domains)):
            for j in range(i + 1, len(mat_domains)):
                dom_a, dom_b = mat_domains[i], mat_domains[j]
                avg_surf_pts = int(round(0.5 * (dom_a.training_config.number_of_surface_points + dom_b.training_config.number_of_surface_points)))
                mult = 0.5 * (float(getattr(dom_a.training_config, 'sampling_density_multiplier', 1.0)) + float(getattr(dom_b.training_config, 'sampling_density_multiplier', 1.0)))
                actual_pts = int(round(avg_surf_pts * mult))

                shared_pts = self.sample_shared_interface_points(dom_a, dom_b, actual_pts)
                if shared_pts.shape[0] > 0:
                    pair_key = f"domain_{dom_a.domain_id}_vs_domain_{dom_b.domain_id}"
                    domains_data["shared_interfaces"][pair_key] = {
                        "domain_a": dom_a,
                        "domain_b": dom_b,
                        "key_a": f"domain_{dom_a.domain_id}",
                        "key_b": f"domain_{dom_b.domain_id}",
                        "interface_points": shared_pts.detach().clone().requires_grad_(True)
                    }

        return domains_data
'''

    # =========================================================================
    # 2. CẬP NHẬT: physics_domain/physical_equations/interface_loss.py
    # (Loại bỏ tính toán SDF động, tính Loss trực tiếp trên tập điểm đã chia trước)
    # =========================================================================
    files_to_update[os.path.join(equations_dir, 'interface_loss.py')] = '''import torch

class InterfaceLoss:
    def __init__(self, electro_magnetic_pinn=None, L0=0.1, H0=800000.0, nu0=795774.715459):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        self._L0 = L0
        self._H0 = H0
        self._nu0 = nu0

    @property
    def L0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.L0
        return self._L0

    @property
    def H0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.H0
        return self._H0

    @property
    def nu0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.nu0
        return self._nu0

    def _compute_normals_and_tangents_from_outline(self, domain, surf_pts):
        start_pts = domain.precomputed_edge_starts.to(device=surf_pts.device, dtype=surf_pts.dtype)
        edge_vecs = domain.precomputed_edge_vecs.to(device=surf_pts.device, dtype=surf_pts.dtype)
        edge_lens = domain.precomputed_edge_lens.to(device=surf_pts.device, dtype=surf_pts.dtype)
        unit_tangents = domain.precomputed_unit_tangents.to(device=surf_pts.device, dtype=surf_pts.dtype)
        unit_normals = domain.precomputed_unit_normals.to(device=surf_pts.device, dtype=surf_pts.dtype)

        pt_to_start = surf_pts.unsqueeze(1) - start_pts.unsqueeze(0)
        denom = (edge_lens.squeeze(1) ** 2 + 1e-12).unsqueeze(0)
        proj = torch.clamp(torch.sum(pt_to_start * edge_vecs.unsqueeze(0), dim=2) / denom, 0.0, 1.0)
        closest = start_pts.unsqueeze(0) + proj.unsqueeze(-1) * edge_vecs.unsqueeze(0)
        dist_sq = torch.sum((surf_pts.unsqueeze(1) - closest) ** 2, dim=2)
        edge_idx = torch.argmin(dist_sq, dim=1)

        t_edge = torch.gather(proj, 1, edge_idx.unsqueeze(1)).detach()
        corner_w = (torch.tanh(12.0 * t_edge) * torch.tanh(12.0 * (1.0 - t_edge))).detach()

        if self.electro_magnetic_pinn is not None:
            sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
            mat_verts = [
                d.outline_tensor.to(device=surf_pts.device, dtype=surf_pts.dtype)
                for d in sim_domain.domains_list
                if getattr(d, 'domain_name', '') != 'background' and d.outline_tensor is not None
            ]
            if mat_verts:
                all_verts = torch.cat(mat_verts, dim=0)
                v_dists = torch.cdist(surf_pts.detach(), all_verts, p=2.0)
                min_v_dist, _ = torch.min(v_dists, dim=1, keepdim=True)
                vertex_w = (torch.tanh(25.0 * (min_v_dist / self.L0)) ** 2).detach()
                corner_w = corner_w * vertex_w

        nx = unit_normals[edge_idx, 0:1].detach()
        ny = unit_normals[edge_idx, 1:2].detach()
        tx = unit_tangents[edge_idx, 0:1].detach()
        ty = unit_tangents[edge_idx, 1:2].detach()
        return nx, ny, tx, ty, corner_w

    def _compute_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            zero = torch.tensor(0.0, device=pts.device)
            return zero, zero, zero, zero

        model = self.electro_magnetic_pinn.pinn_architecture_instance
        nx, ny, tx, ty, corner_w = self._compute_normals_and_tangents_from_outline(domain_a, pts)

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            _, _, _, _, corner_w_b = self._compute_normals_and_tangents_from_outline(domain_b, pts)
            corner_w = torch.minimum(corner_w, corner_w_b)

        Az_a = model.forward_network(key_a, pts)
        Az_b = model.forward_network(key_b, pts)

        loss_val = torch.mean((Az_a - Az_b) ** 2)

        grad_a = torch.autograd.grad(
            outputs=Az_a,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_a),
            create_graph=True
        )[0]

        grad_b = torch.autograd.grad(
            outputs=Az_b,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_b),
            create_graph=True
        )[0]

        Bx_a_star = grad_a[:, 1:2] * self.L0
        By_a_star = -grad_a[:, 0:1] * self.L0
        Bx_b_star = grad_b[:, 1:2] * self.L0
        By_b_star = -grad_b[:, 0:1] * self.L0

        Bn_a_star = Bx_a_star * nx + By_a_star * ny
        Bn_b_star = Bx_b_star * nx + By_b_star * ny
        loss_Bn = torch.mean(corner_w * ((Bn_a_star - Bn_b_star) ** 2))

        B_mag_a_phys = torch.sqrt(Bx_a_star**2 + By_a_star**2 + 1e-12) * (self.H0 / self.nu0)
        nu_a_star = domain_a.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_a_phys) / self.nu0
        hx_a, hy_a = domain_a.evaluate_magnetization_vector(pts)
        hx_a_star = hx_a / self.H0
        hy_a_star = hy_a / self.H0

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            B_mag_b_phys = torch.sqrt(Bx_b_star**2 + By_b_star**2 + 1e-12) * (self.H0 / self.nu0)
            nu_b_star = domain_b.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_b_phys) / self.nu0
            hx_b, hy_b = domain_b.evaluate_magnetization_vector(pts)
            hx_b_star = hx_b / self.H0
            hy_b_star = hy_b / self.H0
        else:
            nu_b_star = torch.ones_like(nu_a_star)
            hx_b_star = torch.zeros_like(hx_a_star)
            hy_b_star = torch.zeros_like(hy_a_star)

        Hx_a_star = nu_a_star * Bx_a_star - hx_a_star
        Hy_a_star = nu_a_star * By_a_star - hy_a_star
        Hx_b_star = nu_b_star * Bx_b_star - hx_b_star
        Hy_b_star = nu_b_star * By_b_star - hy_b_star

        Ht_a_star = Hx_a_star * tx + Hy_a_star * ty
        Ht_b_star = Hx_b_star * tx + Hy_b_star * ty
        nu_pair_scale = torch.clamp(torch.maximum(nu_a_star.detach(), nu_b_star.detach()), min=1e-6)
        ht_diff_normalized = (Ht_a_star - Ht_b_star) / nu_pair_scale
        loss_Ht = torch.mean(corner_w * (ht_diff_normalized ** 2))

        w_az = float(getattr(domain_a.training_config, 'interface_potential_weight', 1000.0))
        w_bn = float(getattr(domain_a.training_config, 'interface_bn_weight', 500.0))
        w_ht = float(getattr(domain_a.training_config, 'interface_ht_weight', 500.0))

        weighted_interface = w_az * loss_val + w_bn * loss_Bn + w_ht * loss_Ht
        return weighted_interface, loss_val, loss_Bn, loss_Ht

    def compute_loss_components(self, network_key, domain, surf_pts, background_key='background', contact_tol=1e-4):
        bg_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance.background_domain
        return self._compute_pair_components(network_key, domain, background_key, bg_domain, surf_pts)

    def compute_loss(self, network_key, domain, surf_pts, background_key='background'):
        weighted_interface, _, _, _ = self.compute_loss_components(network_key, domain, surf_pts, background_key)
        return weighted_interface
'''

    # =========================================================================
    # 3. CẬP NHẬT: training_engine/training_manager.py
    # (Hỗ trợ cấu trúc domains_data chia trước cả Outer và Shared Interfaces)
    # =========================================================================
    files_to_update[os.path.join(training_dir, 'training_manager.py')] = '''import time
import torch
import torch.optim as optim

class TrainingManager:
    def __init__(self, electro_magnetic_pinn):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        
        t_cfg = self.electro_magnetic_pinn.training_config
        self.base_learning_rate_adam = t_cfg.learning_rate_adam
        self.target_loss = t_cfg.target_loss
        self.lbfgs_learning_rate = t_cfg.lbfgs_learning_rate
        self.lbfgs_maximum_iterations = t_cfg.lbfgs_maximum_iterations
        self.lbfgs_maximum_evaluations = t_cfg.lbfgs_maximum_evaluations
        self.lbfgs_tolerance_gradient = t_cfg.lbfgs_tolerance_gradient
        self.lbfgs_tolerance_change = t_cfg.lbfgs_tolerance_change
        self.lbfgs_history_size = t_cfg.lbfgs_history_size
        
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        self.optimizer_adam = optim.Adam(model.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=self.lbfgs_maximum_iterations,
            max_eval=self.lbfgs_maximum_evaluations,
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )
        self.last_loss_breakdown = {}
        self.completed_adam_epochs = 0

    @property
    def model(self):
        return self.electro_magnetic_pinn.pinn_architecture_instance

    @property
    def pde_evaluator(self):
        return self.electro_magnetic_pinn.maxwell_pde_loss_instance

    @property
    def interface_evaluator(self):
        return self.electro_magnetic_pinn.interface_loss_instance

    @property
    def visualizer(self):
        return self.electro_magnetic_pinn.visualizer_instance

    @property
    def visualizer_update_interval(self):
        return self.electro_magnetic_pinn.visualization_config.update_interval

    def compute_loss(self, domains_data):
        device = next(self.model.parameters()).device
        dummy_param = next(self.model.parameters())
        total_loss = (dummy_param * 0.0).sum()

        raw_pde_lin_sum = 0.0
        raw_pde_nonlin_sum = 0.0
        weighted_pde_sum = 0.0
        az_if_sum = 0.0
        bn_if_sum = 0.0
        ht_if_sum = 0.0
        weighted_if_sum = 0.0

        if "domains" in domains_data:
            domains_dict = domains_data["domains"]
            shared_dict = domains_data.get("shared_interfaces", {})
        else:
            domains_dict = domains_data
            shared_dict = {}

        bg_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance.background_domain

        # 1. PDE Loss & Outer Interface Loss
        for key, data in domains_dict.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)
                    
                A_z_star = self.model.forward_network(key, pts)

                grad_A_star = torch.autograd.grad(outputs=A_z_star, inputs=pts, grad_outputs=torch.ones_like(A_z_star), create_graph=True)[0]
                Bx_star = grad_A_star[:, 1:2] * self.pde_evaluator.L0
                By_star = -grad_A_star[:, 0:1] * self.pde_evaluator.L0
                B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * (self.pde_evaluator.H0 / self.pde_evaluator.nu0)

                nu_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys).to(device)
                nu_lin_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)

                hx, hy = domain.evaluate_magnetization_vector(pts)
                hx, hy = hx.to(device), hy.to(device)
                jz = domain.evaluate_current_density(pts).to(device)
                
                res_lin, res_nonlin = self.pde_evaluator.compute_residual_components(
                    xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_linear_phys=nu_lin_phys
                )
                fading_w = self.pde_evaluator.compute_fading_weight(domain, pts)
                
                loss_lin = torch.mean((res_lin ** 2) * fading_w)
                loss_nonlin = torch.mean((res_nonlin ** 2) * fading_w)

                w_pde = float(domain.training_config.pde_weight)
                w_pde_nonlin = float(getattr(domain.training_config, 'pde_nonlinear_weight', 1.0))

                pde_domain_loss = w_pde * loss_lin + w_pde_nonlin * loss_nonlin
                total_loss = total_loss + pde_domain_loss

                raw_pde_lin_sum += loss_lin.item()
                raw_pde_nonlin_sum += loss_nonlin.item()
                weighted_pde_sum += pde_domain_loss.item()

            outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
            if outer_pts.shape[0] > 0 and key != 'background':
                if not outer_pts.requires_grad:
                    outer_pts = outer_pts.detach().clone().requires_grad_(True)
                    
                w_if_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=key,
                    domain_a=domain,
                    key_b='background',
                    domain_b=bg_domain,
                    pts=outer_pts
                )
                scaled_if_loss = float(domain.training_config.interface_weight) * w_if_loss
                total_loss = total_loss + scaled_if_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_if_loss.item()

        # 2. Shared Interface Loss (Ghép trực tiếp Mạng A vs Mạng B)
        for pair_key, pair_data in shared_dict.items():
            pts = pair_data["interface_points"].to(device)
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)

                w_shared_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=pair_data["key_a"],
                    domain_a=pair_data["domain_a"],
                    key_b=pair_data["key_b"],
                    domain_b=pair_data["domain_b"],
                    pts=pts
                )
                w_weight = 0.5 * (float(pair_data["domain_a"].training_config.interface_weight) + 
                                  float(pair_data["domain_b"].training_config.interface_weight))
                scaled_shared_loss = w_weight * w_shared_loss
                total_loss = total_loss + scaled_shared_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_shared_loss.item()

        self.last_loss_breakdown = {
            "raw_pde_lin": raw_pde_lin_sum,
            "raw_pde_nonlin": raw_pde_nonlin_sum,
            "weighted_pde": weighted_pde_sum,
            "interface_Az": az_if_sum,
            "interface_Bn": bn_if_sum,
            "interface_Ht": ht_if_sum,
            "weighted_interface": weighted_if_sum,
            "total": total_loss.item()
        }

        return total_loss

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        self.model.train()
        
        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_learning_rate_adam
            param_group['lr'] = self.base_learning_rate_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=max(epochs, 1), eta_min=1e-6)
        stop_reason = f"Hoàn thành toàn bộ {epochs} epochs."
        non_finite_streak = 0
        self.completed_adam_epochs = 0
        
        for epoch in range(epochs):
            if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
                domains_data = resample_fn()

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            
            if not torch.isfinite(loss):
                non_finite_streak += 1
                if non_finite_streak >= 5:
                    stop_reason = "Dừng sớm do Loss không hữu hạn (NaN/Inf) liên tiếp 5 lần."
                    break
                continue
            non_finite_streak = 0
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            self.completed_adam_epochs = epoch + 1
            current_loss_value = loss.item()
            
            if (epoch + 1) == 1 or (epoch + 1) % 100 == 0:
                bd = self.last_loss_breakdown
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE(Lin/NonLin) = {bd.get('raw_pde_lin', 0.0):.3e}/{bd.get('raw_pde_nonlin', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
                )
                
            if self.visualizer and (epoch + 1) % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
                
        print(f"[*] KẾT THÚC ADAM. Lý do: {stop_reason}")
        return domains_data

    def train_lbfgs(self, max_iterations, domains_data):
        if max_iterations <= 0:
            return domains_data

        self.model.train()
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=max_iterations,
            max_eval=int(max_iterations * 1.25),
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )

        lbfgs_counter = [0]
        last_loss = [0.0]
        
        def closure():
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            
            lbfgs_counter[0] += 1
            last_loss[0] = loss.item()
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 25 == 0:
                bd = self.last_loss_breakdown
                print(
                    f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {loss.item():.4e} | "
                    f"PDE(Lin/NonLin) = {bd.get('raw_pde_lin', 0.0):.3e}/{bd.get('raw_pde_nonlin', 0.0):.3e}"
                )
            return loss
            
        self.optimizer_lbfgs.step(closure)
        return domains_data
'''

    # Thực hiện ghi nội dung mới vào các file tương ứng
    for file_path, code_content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        print(f"[*] Đã ghi và cập nhật thành công: {os.path.relpath(file_path, project_root)}")

    print("\n[✔] TẤT CẢ FILE ĐÃ ĐƯỢC NÂNG CẤP SANG CHIẾN LƯỢC PRE-PARTITIONED SHARED INTERFACES!")

if __name__ == "__main__":
    apply_shared_interface_prepartition_updates()