import os
import sys

def modify_project():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    pde_loss_path = os.path.join(project_root, 'physics_domain', 'physical_equations', 'maxwell_pde_loss.py')
    interface_loss_path = os.path.join(project_root, 'physics_domain', 'physical_equations', 'interface_loss.py')
    sampler_path = os.path.join(project_root, 'physics_domain', 'collocation_sampler.py')
    test_sim4_path = os.path.join(project_root, 'tests', 'test_simulation4.py')

    pde_loss_content = '''import torch

class MaxwellPDELoss:
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

    def compute_residual(self, xy, A_z_star, nu, J_z, H_cx, H_cy):
        nu_star = nu / self.nu0
        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)
        
        grad_A_star = torch.autograd.grad(
            outputs=A_z_star,
            inputs=xy,
            grad_outputs=torch.ones_like(A_z_star),
            create_graph=True
        )[0]
        
        dAz_star_dx_star = grad_A_star[:, 0:1] * self.L0
        dAz_star_dy_star = grad_A_star[:, 1:2] * self.L0
        
        H_x_star = nu_star * dAz_star_dy_star - H_cx_star
        H_y_star = -nu_star * dAz_star_dx_star - H_cy_star
        
        grad_Hx_star = torch.autograd.grad(
            outputs=H_x_star,
            inputs=xy,
            grad_outputs=torch.ones_like(H_x_star),
            create_graph=True
        )[0]
        dHx_star_dy_star = grad_Hx_star[:, 1:2] * self.L0
        
        grad_Hy_star = torch.autograd.grad(
            outputs=H_y_star,
            inputs=xy,
            grad_outputs=torch.ones_like(H_y_star),
            create_graph=True
        )[0]
        dHy_star_dx_star = grad_Hy_star[:, 0:1] * self.L0
        
        residual_star = dHy_star_dx_star - dHx_star_dy_star - J_z_star
        nu_scale = torch.clamp(nu_star.detach(), min=1e-6)
        return residual_star / nu_scale

    def compute_fading_weight(self, domain, xy):
        beta = float(getattr(domain.training_config, 'pde_fading_beta', 150.0))
        if beta <= 0.0:
            return torch.ones((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)

        if self.electro_magnetic_pinn is not None:
            sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
            mat_verts = [
                d.outline_tensor.to(device=xy.device, dtype=xy.dtype)
                for d in sim_domain.domains_list
                if getattr(d, 'domain_name', '') != 'background' and d.outline_tensor is not None
            ]
            if mat_verts:
                all_verts = torch.cat(mat_verts, dim=0)
                dists = torch.cdist(xy.detach(), all_verts, p=2.0)
                min_corner_dist, _ = torch.min(dists, dim=1, keepdim=True)
                corner_dist_star = min_corner_dist / self.L0
                return 1.0 - torch.exp(-beta * corner_dist_star)

        sdf = domain.compute_signed_distance_field(xy).view(-1, 1)
        abs_sdf_star = torch.abs(sdf.detach()) / self.L0
        return 1.0 - torch.exp(-beta * abs_sdf_star)
'''

    interface_loss_content = '''import torch

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
        vertices = domain.outline_tensor.to(device=surf_pts.device, dtype=surf_pts.dtype)
        start_pts = vertices
        end_pts = torch.roll(vertices, shifts=-1, dims=0)
        edge_vecs = end_pts - start_pts
        edge_lens = torch.sqrt(torch.sum(edge_vecs ** 2, dim=1, keepdim=True) + 1e-12)

        signed_area = 0.5 * torch.sum(start_pts[:, 0] * end_pts[:, 1] - end_pts[:, 0] * start_pts[:, 1])
        orient = 1.0 if signed_area.item() >= 0.0 else -1.0

        unit_tangents = orient * (edge_vecs / edge_lens)
        unit_normals = torch.cat([unit_tangents[:, 1:2], -unit_tangents[:, 0:1]], dim=1)

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
            return zero, zero, zero

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

        nu_a_star = domain_a.evaluate_reluctivity(pts) / self.nu0
        hx_a, hy_a = domain_a.evaluate_magnetization_vector(pts)
        hx_a_star = hx_a / self.H0
        hy_a_star = hy_a / self.H0

        if domain_b is not None:
            nu_b_star = domain_b.evaluate_reluctivity(pts) / self.nu0
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

        return loss_val, loss_Bn, loss_Ht

    def compute_loss_components(self, network_key, domain, surf_pts, background_key='background', contact_tol=1e-4):
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        bg_domain = sim_domain.background_domain
        other_domains = [
            d for d in sim_domain.domains_list
            if d is not domain and getattr(d, 'domain_name', '') != 'background'
        ]

        nx_a, ny_a, _, _, _ = self._compute_normals_and_tangents_from_outline(domain, surf_pts)

        remaining_mask = torch.ones(surf_pts.shape[0], dtype=torch.bool, device=surf_pts.device)
        total_val = torch.tensor(0.0, device=surf_pts.device)
        total_bn = torch.tensor(0.0, device=surf_pts.device)
        total_ht = torch.tensor(0.0, device=surf_pts.device)

        for other_dom in other_domains:
            if not remaining_mask.any():
                break
            other_sdf = other_dom.compute_signed_distance_field(surf_pts).view(-1)
            nx_b, ny_b, _, _, _ = self._compute_normals_and_tangents_from_outline(other_dom, surf_pts)
            normal_dot = (nx_a * nx_b + ny_a * ny_b).view(-1)
            contact_mask = remaining_mask & (torch.abs(other_sdf) <= contact_tol) & (normal_dot < -0.85)
            if contact_mask.any():
                other_key = f"domain_{other_dom.domain_id}"
                v_l, bn_l, ht_l = self._compute_pair_components(
                    network_key, domain, other_key, other_dom, surf_pts[contact_mask]
                )
                ratio = float(contact_mask.sum().item()) / float(surf_pts.shape[0])
                total_val = total_val + ratio * v_l
                total_bn = total_bn + ratio * bn_l
                total_ht = total_ht + ratio * ht_l
                remaining_mask = remaining_mask & (~contact_mask)

        if remaining_mask.any():
            v_bg, bn_bg, ht_bg = self._compute_pair_components(
                network_key, domain, background_key, bg_domain, surf_pts[remaining_mask]
            )
            ratio = float(remaining_mask.sum().item()) / float(surf_pts.shape[0])
            total_val = total_val + ratio * v_bg
            total_bn = total_bn + ratio * bn_bg
            total_ht = total_ht + ratio * ht_bg

        w_az = float(getattr(domain.training_config, 'interface_potential_weight', 1000.0))
        w_bn = float(getattr(domain.training_config, 'interface_bn_weight', 500.0))
        w_ht = float(getattr(domain.training_config, 'interface_ht_weight', 500.0))

        weighted_interface = w_az * total_val + w_bn * total_bn + w_ht * total_ht
        return weighted_interface, total_val, total_bn, total_ht

    def compute_loss(self, network_key, domain, surf_pts, background_key='background'):
        weighted_interface, _, _, _ = self.compute_loss_components(network_key, domain, surf_pts, background_key)
        return weighted_interface
'''

    sampler_content = '''import torch
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
'''

    test_sim4_content = '''import os
import sys
import shutil

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation_domain_engine.material_domain.material_domain import MaterialDomain
from electro_magnetic_pinn import ElectroMagneticPINN

def main():
    output_dir = "figure_iron_yoke"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print("version 2.0 - Dual Magnets on Constant Permeability Iron Yoke Simulation (Fixed Multi-Material Interface)")
    
    model = ElectroMagneticPINN()
    
    max_coercive_field = 800000.0
    model.simulation_domain_engine_instance.physics_config.scale_H0 = max_coercive_field * 1.2
    model.simulation_domain_engine_instance.physics_config.scale_L0 = 0.1 * 1.2
    
    model.simulation_domain_engine_instance.x_boundaries_tuple = (-0.1, 0.1)
    model.simulation_domain_engine_instance.y_boundaries_tuple = (-0.1, 0.1)
    
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_before_lbfgs = True
    
    model.training_config.epochs_adam = 4000
    model.training_config.epochs_lbfgs = 2000
    model.training_config.learning_rate_adam = 1e-3
    model.training_config.target_loss = 1e-7
    
    model.training_config.lbfgs_learning_rate = 0.8
    model.training_config.lbfgs_maximum_iterations = 2000
    model.training_config.lbfgs_maximum_evaluations = 2500
    model.training_config.lbfgs_tolerance_gradient = 1e-8
    model.training_config.lbfgs_tolerance_change = 1e-10
    model.training_config.lbfgs_history_size = 80
    
    model.visualization_config.active = True
    model.visualization_config.update_interval = 50
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "iron_yoke_training_process.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "iron_yoke_training_process.mp4"
    model.visualization_config.animation_fps = 5
    model.visualization_config.gif_fps = 5

    iron_yoke = MaterialDomain(domain_name="iron_yoke")
    iron_yoke.geometry_config.outline = [
        [-0.06, -0.03], 
        [0.06, -0.03], 
        [0.06, 0.0], 
        [-0.06, 0.0]
    ]
    iron_yoke.physics_config.relative_permeability = 1000.0
    iron_yoke.physics_config.coercive = [0.0, 0.0]
    iron_yoke.physics_config.current = 0.0
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_weight = 1.0
    iron_yoke.training_config.interface_potential_weight = 1000.0
    iron_yoke.training_config.interface_bn_weight = 400.0
    iron_yoke.training_config.interface_ht_weight = 400.0
    iron_yoke.training_config.pde_fading_beta = 80.0
    iron_yoke.training_config.number_of_interior_points = 3000
    iron_yoke.training_config.number_of_surface_points = 1600
    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128
    iron_yoke.build_domain()

    magnet_left = MaterialDomain(domain_name="magnet_left")
    magnet_left.geometry_config.outline = [
        [-0.05, 0.0], 
        [-0.005, 0.0], 
        [-0.005, 0.018], 
        [-0.05, 0.018]
    ]
    magnet_left.physics_config.relative_permeability = 1.05
    magnet_left.physics_config.coercive = [0.0, 800000.0]
    magnet_left.physics_config.current = 0.0
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_weight = 1.0
    magnet_left.training_config.interface_potential_weight = 1000.0
    magnet_left.training_config.interface_bn_weight = 400.0
    magnet_left.training_config.interface_ht_weight = 400.0
    magnet_left.training_config.pde_fading_beta = 80.0
    magnet_left.training_config.number_of_interior_points = 2000
    magnet_left.training_config.number_of_surface_points = 1200
    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128
    magnet_left.build_domain()

    magnet_right = MaterialDomain(domain_name="magnet_right")
    magnet_right.geometry_config.outline = [
        [0.005, 0.0], 
        [0.05, 0.0], 
        [0.05, 0.018], 
        [0.005, 0.018]
    ]
    magnet_right.physics_config.relative_permeability = 1.05
    magnet_right.physics_config.coercive = [0.0, -800000.0]
    magnet_right.physics_config.current = 0.0
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_weight = 1.0
    magnet_right.training_config.interface_potential_weight = 1000.0
    magnet_right.training_config.interface_bn_weight = 400.0
    magnet_right.training_config.interface_ht_weight = 400.0
    magnet_right.training_config.pde_fading_beta = 80.0
    magnet_right.training_config.number_of_interior_points = 2000
    magnet_right.training_config.number_of_surface_points = 1200
    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128
    magnet_right.build_domain()

    model.simulation_domain_engine_instance.add_material_domain(iron_yoke)
    model.simulation_domain_engine_instance.add_material_domain(magnet_left)
    model.simulation_domain_engine_instance.add_material_domain(magnet_right)

    bg_training_config = type('', (), {})()
    bg_training_config.pde_weight = 1.0
    bg_training_config.interface_weight = 1.0
    bg_training_config.pde_fading_beta = 80.0
    bg_training_config.sampling_density_multiplier = 1.0
    bg_training_config.number_of_interior_points = 8000
    bg_training_config.number_of_surface_points = 0

    bg_network_config = type('', (), {})()
    bg_network_config.hidden_layers = 4
    bg_network_config.hidden_neurons = 64
    bg_network_config.activation_function = nn.SiLU()
    bg_network_config.fourier_features = True
    bg_network_config.fourier_sigma = 1.2
    bg_network_config.fourier_mapping_size = 128

    model.simulation_domain_engine_instance.build_background_domain(
        domain_name="background",
        training_config=bg_training_config,
        network_config=bg_network_config
    )

    model.update_electromagnetic_pinn()

    xb = model.simulation_domain_engine_instance.x_boundaries_tuple
    yb = model.simulation_domain_engine_instance.y_boundaries_tuple
    
    model.simulation_domain_engine_instance.plot_problem_definition(resolution=100)

    model.execute_training_process()
    
    resolution = 180
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(xy_points_tensor)
    assert torch.all(torch.isfinite(A_z_pred)) and torch.all(torch.isfinite(B_x_pred)) and torch.all(torch.isfinite(B_y_pred)), "Trường từ tính toán không hữu hạn."
    
    A_z_grid = A_z_pred.cpu().numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.cpu().numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.cpu().numpy().reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)
    
    fig1, axs = plt.subplots(2, 2, figsize=(12, 10))
    contour_az = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=60, cmap="jet")
    fig1.colorbar(contour_az, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
    axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
    axs[0, 0].set_aspect('equal')
    
    b_max_val = max(float(np.max(B_mag_grid)), 1e-3)
    levels_b = np.linspace(0.0, b_max_val, 60)
    contour_b = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=levels_b, cmap="jet", vmin=0.0, vmax=b_max_val)
    fig1.colorbar(contour_b, ax=axs[0, 1], label=r"$|B|$ (T)")
    axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_aspect('equal')
    
    contour_bx = axs[1, 0].contourf(X_grid, Y_grid, B_x_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_bx, ax=axs[1, 0], label=r"$B_x$ (T)")
    axs[1, 0].set_title(r"Magnetic Field Component ($B_x$)")
    axs[1, 0].set_aspect('equal')
    
    contour_by = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig1.colorbar(contour_by, ax=axs[1, 1], label=r"$B_y$ (T)")
    axs[1, 1].set_title(r"Magnetic Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')
    fig1.tight_layout()
    plt.savefig(f'{output_dir}/final_fields_grid.png', dpi=150)
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    contour_b_bg = ax2.contourf(X_grid, Y_grid, B_mag_grid, levels=levels_b, cmap="jet", alpha=0.35, vmin=0.0, vmax=b_max_val)
    fig2.colorbar(contour_b_bg, ax=ax2, label=r"$|B|$ (T)")
    
    step = 6
    B_sub = np.maximum(B_mag_grid[::step, ::step], 1e-12)
    ax2.quiver(
        X_grid[::step, ::step], Y_grid[::step, ::step],
        B_x_grid[::step, ::step] / B_sub, B_y_grid[::step, ::step] / B_sub,
        scale=35.0, color='black', pivot='mid'
    )
    ax2.set_title(r"Magnetic Flux Density Vectors ($\mathbf{B}$)")
    ax2.set_aspect('equal')
    fig2.tight_layout()
    
    plt.savefig(f'{output_dir}/final_results.png', dpi=150)
    plt.close('all')
    print(f'[*] Đã lưu kết quả mô phỏng vào {output_dir}/final_results.png và {output_dir}/final_fields_grid.png')

if __name__ == "__main__":
    main()
'''

    with open(pde_loss_path, 'w', encoding='utf-8') as f:
        f.write(pde_loss_content)
    print(f"[✓] Đã cập nhật: {pde_loss_path}")

    with open(interface_loss_path, 'w', encoding='utf-8') as f:
        f.write(interface_loss_content)
    print(f"[✓] Đã cập nhật: {interface_loss_path}")

    with open(sampler_path, 'w', encoding='utf-8') as f:
        f.write(sampler_content)
    print(f"[✓] Đã cập nhật: {sampler_path}")

    with open(test_sim4_path, 'w', encoding='utf-8') as f:
        f.write(test_sim4_content)
    print(f"[✓] Đã cập nhật: {test_sim4_path}")

    print("[*] Quá trình modify_project đã hoàn tất thành công.")

if __name__ == '__main__':
    modify_project()