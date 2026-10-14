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
        return residual_star

    def compute_fading_weight(self, domain, xy):
        beta = float(getattr(domain.training_config, 'pde_fading_beta', 150.0))
        if beta <= 0.0:
            return torch.ones((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)

        if getattr(domain, 'domain_name', '') == 'background' and self.electro_magnetic_pinn is not None:
            sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
            sdf = sim_domain.compute_global_signed_distance_field(xy).view(-1, 1)
        else:
            sdf = domain.compute_signed_distance_field(xy).view(-1, 1)

        abs_sdf_star = torch.abs(sdf.detach()) / self.L0
        return 1.0 - torch.exp(-beta * abs_sdf_star)
