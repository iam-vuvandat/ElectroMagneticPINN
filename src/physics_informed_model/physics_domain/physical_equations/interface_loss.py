import torch

class InterfaceLoss:
    def __init__(self, electro_magnetic_pinn=None, L0=0.05, H0=1200000.0, nu0=795774.715459):
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

    def _compute_edge_unit_tangents(self, domain, surf_pts):
        vertices = domain.outline_tensor.to(device=surf_pts.device, dtype=surf_pts.dtype)
        start_pts = vertices
        end_pts = torch.roll(vertices, shifts=-1, dims=0)
        edge_vecs = end_pts - start_pts
        edge_lens = torch.sqrt(torch.sum(edge_vecs ** 2, dim=1, keepdim=True) + 1e-12)
        unit_tangents = edge_vecs / edge_lens

        pt_to_start = surf_pts.unsqueeze(1) - start_pts.unsqueeze(0)
        denom = (edge_lens.squeeze(1) ** 2 + 1e-12).unsqueeze(0)
        proj = torch.clamp(torch.sum(pt_to_start * edge_vecs.unsqueeze(0), dim=2) / denom, 0.0, 1.0)
        closest = start_pts.unsqueeze(0) + proj.unsqueeze(-1) * edge_vecs.unsqueeze(0)
        dist_sq = torch.sum((surf_pts.unsqueeze(1) - closest) ** 2, dim=2)
        edge_idx = torch.argmin(dist_sq, dim=1)

        tx = unit_tangents[edge_idx, 0:1].detach()
        ty = unit_tangents[edge_idx, 1:2].detach()
        return tx, ty

    def compute_loss(self, network_key, domain, surf_pts, background_key='background'):
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        bg_domain = sim_domain.background_domain

        Az_mat = model.forward_network(network_key, surf_pts)
        Az_bg = model.forward_network(background_key, surf_pts)

        potential_continuity_loss = torch.mean((Az_mat - Az_bg) ** 2)

        grad_Az_mat = torch.autograd.grad(
            outputs=Az_mat,
            inputs=surf_pts,
            grad_outputs=torch.ones_like(Az_mat),
            create_graph=True,
            retain_graph=True
        )[0]

        grad_Az_bg = torch.autograd.grad(
            outputs=Az_bg,
            inputs=surf_pts,
            grad_outputs=torch.ones_like(Az_bg),
            create_graph=True,
            retain_graph=True
        )[0]

        dAz_mat_dx_star = grad_Az_mat[:, 0:1] * self.L0
        dAz_mat_dy_star = grad_Az_mat[:, 1:2] * self.L0
        dAz_bg_dx_star = grad_Az_bg[:, 0:1] * self.L0
        dAz_bg_dy_star = grad_Az_bg[:, 1:2] * self.L0

        tx, ty = self._compute_edge_unit_tangents(domain, surf_pts)

        bn_continuity_loss = torch.mean(((dAz_mat_dx_star - dAz_bg_dx_star) * tx + (dAz_mat_dy_star - dAz_bg_dy_star) * ty) ** 2)

        nu_mat_star = domain.evaluate_reluctivity(surf_pts) / self.nu0
        hx_mat, hy_mat = domain.evaluate_magnetization_vector(surf_pts)
        hx_mat_star = hx_mat / self.H0
        hy_mat_star = hy_mat / self.H0

        if bg_domain is not None:
            nu_bg_star = bg_domain.evaluate_reluctivity(surf_pts) / self.nu0
            hx_bg, hy_bg = bg_domain.evaluate_magnetization_vector(surf_pts)
            hx_bg_star = hx_bg / self.H0
            hy_bg_star = hy_bg / self.H0
        else:
            nu_bg_star = 1.0
            hx_bg_star = 0.0
            hy_bg_star = 0.0

        Hx_mat_star = nu_mat_star * dAz_mat_dy_star - hx_mat_star
        Hy_mat_star = -nu_mat_star * dAz_mat_dx_star - hy_mat_star

        Hx_bg_star = nu_bg_star * dAz_bg_dy_star - hx_bg_star
        Hy_bg_star = -nu_bg_star * dAz_bg_dx_star - hy_bg_star

        Ht_mat_star = Hx_mat_star * tx + Hy_mat_star * ty
        Ht_bg_star = Hx_bg_star * tx + Hy_bg_star * ty

        ht_continuity_loss = torch.mean((Ht_mat_star - Ht_bg_star) ** 2)

        return potential_continuity_loss + bn_continuity_loss + ht_continuity_loss
