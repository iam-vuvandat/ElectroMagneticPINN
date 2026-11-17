import torch

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

    def compute_first_order_pde_residuals(self, xy, field_star, nu_phys, J_z, H_cx, H_cy, nu_star_net=None):
        if nu_star_net is not None:
            nu_star = nu_star_net
        else:
            nu_star = nu_phys / self.nu0

        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)

        Bx_star = field_star[:, 0:1]
        By_star = field_star[:, 1:2]
        Hx_star = nu_star * Bx_star - H_cx_star
        Hy_star = nu_star * By_star - H_cy_star

        grad_Bx = torch.autograd.grad(
            outputs=Bx_star,
            inputs=xy,
            grad_outputs=torch.ones_like(Bx_star),
            create_graph=True
        )[0]
        grad_By = torch.autograd.grad(
            outputs=By_star,
            inputs=xy,
            grad_outputs=torch.ones_like(By_star),
            create_graph=True
        )[0]
        grad_Hx = torch.autograd.grad(
            outputs=Hx_star,
            inputs=xy,
            grad_outputs=torch.ones_like(Hx_star),
            create_graph=True
        )[0]
        grad_Hy = torch.autograd.grad(
            outputs=Hy_star,
            inputs=xy,
            grad_outputs=torch.ones_like(Hy_star),
            create_graph=True
        )[0]

        dBx_dx_star = grad_Bx[:, 0:1] * self.L0
        dBy_dy_star = grad_By[:, 1:2] * self.L0
        dHx_dy_star = grad_Hx[:, 1:2] * self.L0
        dHy_dx_star = grad_Hy[:, 0:1] * self.L0

        nu_scale = torch.clamp(nu_star.detach(), min=1e-6)

        res_div = dBx_dx_star + dBy_dy_star
        res_curl = (dHy_dx_star - dHx_dy_star - J_z_star) / nu_scale
        res_const_x = torch.zeros_like(res_div)
        res_const_y = torch.zeros_like(res_div)

        return res_div, res_curl, res_const_x, res_const_y

    def compute_fading_weight(self, domain, xy):
        beta = float(getattr(domain.training_config, 'pde_fading_beta', 80.0))
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