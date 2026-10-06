import torch

class MaxwellPDELoss:
    def __init__(self, L0=0.05, H0=800000.0, nu0=795774.715459):
        self.L0 = L0
        self.H0 = H0
        self.nu0 = nu0

    def compute_residual(self, xy, A_z_star, nu, J_z, H_cx, H_cy):
        nu_star = nu / self.nu0
        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)
        
        grad_A_star = torch.autograd.grad(
            outputs=A_z_star,
            inputs=xy,
            grad_outputs=torch.ones_like(A_z_star),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dAz_star_dx_star = grad_A_star[:, 0:1] * self.L0
        dAz_star_dy_star = grad_A_star[:, 1:2] * self.L0
        
        H_x_star = nu_star * dAz_star_dy_star - H_cx_star
        H_y_star = -nu_star * dAz_star_dx_star - H_cy_star
        
        grad_Hx_star = torch.autograd.grad(
            outputs=H_x_star,
            inputs=xy,
            grad_outputs=torch.ones_like(H_x_star),
            create_graph=True,
            retain_graph=True
        )[0]
        dHx_star_dy_star = grad_Hx_star[:, 1:2] * self.L0
        
        # BẮT BUỘC retain_graph=True ở đây để đồ thị mạng Nơ-ron sống sót 
        # cho đến khi lệnh loss.backward() ở file training_manager.py được gọi.
        grad_Hy_star = torch.autograd.grad(
            outputs=H_y_star,
            inputs=xy,
            grad_outputs=torch.ones_like(H_y_star),
            create_graph=True,
            retain_graph=True
        )[0]
        dHy_star_dx_star = grad_Hy_star[:, 0:1] * self.L0
        
        residual_star = dHy_star_dx_star - dHx_star_dy_star - J_z_star
        return residual_star
