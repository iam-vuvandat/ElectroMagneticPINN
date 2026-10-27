import torch

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

    def _compute_normals_and_tangents_from_sdf(self, domain, surf_pts, eps=1e-4):
        """
        Tính toán chính xác vectơ pháp tuyến (nx, ny), tiếp tuyến (tx, ty) và trọng số dập mượt góc (corner_w)
        CHO MỌI BIÊN DẠNG PHỨC TẠP (Cong, CSG, CAD) dựa hoàn toàn vào SDF thuần túy và Vi phân Trung tâm.
        """
        with torch.no_grad():
            pts = surf_pts.detach()
            dx = torch.tensor([eps, 0.0], device=pts.device, dtype=pts.dtype)
            dy = torch.tensor([0.0, eps], device=pts.device, dtype=pts.dtype)

            sdf_center = domain.compute_signed_distance_field(pts).view(-1, 1)
            sdf_x_plus = domain.compute_signed_distance_field(pts + dx).view(-1, 1)
            sdf_x_minus = domain.compute_signed_distance_field(pts - dx).view(-1, 1)
            sdf_y_plus = domain.compute_signed_distance_field(pts + dy).view(-1, 1)
            sdf_y_minus = domain.compute_signed_distance_field(pts - dy).view(-1, 1)

            # Gradient bằng vi phân trung tâm 2D
            grad_x = (sdf_x_plus - sdf_x_minus) / (2.0 * eps)
            grad_y = (sdf_y_plus - sdf_y_minus) / (2.0 * eps)

            norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-12)
            nx = (grad_x / norm).detach()
            ny = (grad_y / norm).detach()

            tx = -ny
            ty = nx

            # Tính độ cong Laplacian dập mượt góc nhọn (Corner Damping via SDF Curvature)
            lap_x = (sdf_x_plus - 2.0 * sdf_center + sdf_x_minus) / (eps**2)
            lap_y = (sdf_y_plus - 2.0 * sdf_center + sdf_y_minus) / (eps**2)
            curvature = torch.abs(lap_x + lap_y)

            corner_w = torch.exp(-0.005 * curvature).detach()
            corner_w = torch.clamp(corner_w, min=0.1, max=1.0)

        return nx, ny, tx, ty, corner_w

    def _compute_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            zero = torch.tensor(0.0, device=pts.device)
            return zero, zero, zero, zero

        model = self.electro_magnetic_pinn.pinn_architecture_instance
        nx, ny, tx, ty, corner_w = self._compute_normals_and_tangents_from_sdf(domain_a, pts)

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            _, _, _, _, corner_w_b = self._compute_normals_and_tangents_from_sdf(domain_b, pts)
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
