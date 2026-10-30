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

    def compute_linear_pde_residual(self, xy, A_z_star, nu_phys, J_z, H_cx, H_cy):
        """
        Tính dư số PDE Maxwell-Poisson ở chế độ THUẦN TUYẾN TÍNH:
        curl(H) - Jz = 0  với H = nu * B - Hc
        """
        nu_star = nu_phys / self.nu0
        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)

        # 1. Gradient của Az* -> B*
        grad_A_star = torch.autograd.grad(
            outputs=A_z_star,
            inputs=xy,
            grad_outputs=torch.ones_like(A_z_star),
            create_graph=True
        )[0]

        dAz_dx_star = grad_A_star[:, 0:1] * self.L0
        dAz_dy_star = grad_A_star[:, 1:2] * self.L0

        # 2. H* = nu* * B* - Hc*
        Hx_star = nu_star * dAz_dy_star - H_cx_star
        Hy_star = -nu_star * dAz_dx_star - H_cy_star

        # 3. curl(H*) = dHy*/dx* - dHx*/dy*
        grad_Hx = torch.autograd.grad(outputs=Hx_star, inputs=xy, grad_outputs=torch.ones_like(Hx_star), create_graph=True)[0]
        grad_Hy = torch.autograd.grad(outputs=Hy_star, inputs=xy, grad_outputs=torch.ones_like(Hy_star), create_graph=True)[0]

        dHy_dx_star = grad_Hy[:, 0:1] * self.L0
        dHx_dy_star = grad_Hx[:, 1:2] * self.L0

        res_pde = dHy_dx_star - dHx_dy_star - J_z_star
        nu_scale = torch.clamp(nu_star.detach(), min=1e-6)
        return res_pde / nu_scale

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
