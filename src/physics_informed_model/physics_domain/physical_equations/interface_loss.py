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

    def _compute_normals_and_tangents_from_sdf(self, domain, surf_pts):
        pts_for_n = surf_pts.detach().clone().requires_grad_(True)
        sdf_val = domain.compute_signed_distance_field(pts_for_n).view(-1, 1)
        n_vec = torch.autograd.grad(
            outputs=sdf_val,
            inputs=pts_for_n,
            grad_outputs=torch.ones_like(sdf_val),
            create_graph=False
        )[0]
        n_norm = torch.linalg.norm(n_vec, dim=1, keepdim=True)
        n_unit = (n_vec / (n_norm + 1e-8)).detach()
        nx = n_unit[:, 0:1]
        ny = n_unit[:, 1:2]
        tx = -ny
        ty = nx
        return nx, ny, tx, ty

    def _compute_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            zero = torch.tensor(0.0, device=pts.device)
            return zero, zero, zero

        model = self.electro_magnetic_pinn.pinn_architecture_instance
        nx, ny, tx, ty = self._compute_normals_and_tangents_from_sdf(domain_a, pts)

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
        loss_Bn = torch.mean((Bn_a_star - Bn_b_star) ** 2)

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
            nu_b_star = 1.0
            hx_b_star = 0.0
            hy_b_star = 0.0

        Hx_a_star = nu_a_star * Bx_a_star - hx_a_star
        Hy_a_star = nu_a_star * By_a_star - hy_a_star
        Hx_b_star = nu_b_star * Bx_b_star - hx_b_star
        Hy_b_star = nu_b_star * By_b_star - hy_b_star

        Ht_a_star = Hx_a_star * tx + Hy_a_star * ty
        Ht_b_star = Hx_b_star * tx + Hy_b_star * ty
        loss_Ht = torch.mean((Ht_a_star - Ht_b_star) ** 2)

        return loss_val, loss_Bn, loss_Ht

    def compute_loss_components(self, network_key, domain, surf_pts, background_key='background', contact_tol=1e-4):
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        bg_domain = sim_domain.background_domain
        other_domains = [
            d for d in sim_domain.domains_list
            if d is not domain and getattr(d, 'domain_name', '') != 'background'
        ]

        remaining_mask = torch.ones(surf_pts.shape[0], dtype=torch.bool, device=surf_pts.device)
        total_val = torch.tensor(0.0, device=surf_pts.device)
        total_bn = torch.tensor(0.0, device=surf_pts.device)
        total_ht = torch.tensor(0.0, device=surf_pts.device)

        for other_dom in other_domains:
            if not remaining_mask.any():
                break
            other_sdf = other_dom.compute_signed_distance_field(surf_pts).view(-1)
            contact_mask = remaining_mask & (torch.abs(other_sdf) <= contact_tol)
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
