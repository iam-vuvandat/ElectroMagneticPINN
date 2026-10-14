import os

def fix_autograd_graph_issue():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.abspath(os.path.join(base_dir, '..'))

    training_mgr_file = os.path.join(model_root, 'training_engine', 'training_manager.py')
    forward_net_file = os.path.join(model_root, 'architectures', 'forward_network.py')
    maxwell_loss_file = os.path.join(model_root, 'physics_domain', 'physical_equations', 'maxwell_pde_loss.py')

    # 1. Cập nhật forward_network.py (Đảm bảo B_matrix đồng bộ thiết bị và giữ graph)
    forward_net_content = r'''import torch
import numpy as np

def forward_network(self, network_key, xy):
    if network_key not in self.networks:
        raise ValueError(f"Lỗi: Không tìm thấy mạng con '{network_key}' trong XPINNArchitecture.")
        
    xy_normalized = xy / self.domain_scale
    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == network_key:
            target_domain = domain
            break
        if f"domain_{domain.domain_id}" == network_key:
            target_domain = domain
            break
            
    if target_domain is None:
        raise ValueError(f"Lỗi: Không tìm thấy domain tương ứng với network_key '{network_key}' trong domains_list.")

    config = target_domain.network_config
    
    if config.fourier_features:
        B_matrix = getattr(self, f'B_matrix_{network_key}')
        if B_matrix.device != xy.device:
            B_matrix = B_matrix.to(xy.device)
        projected = (2.0 * np.pi * xy_normalized) @ B_matrix.T
        x_input = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
    else:
        x_input = xy_normalized
        
    raw_output = self.networks[network_key](x_input)
    b_factor = self.boundary_factor(xy)
    A_z_star = raw_output * b_factor
    return A_z_star
'''
    with open(forward_net_file, 'w', encoding='utf-8') as f:
        f.write(forward_net_content)

    # 2. Cập nhật maxwell_pde_loss.py (Đảm bảo create_graph xuyên suốt qua các cấp đạo hàm)
    maxwell_loss_content = r'''import torch

class MaxwellPDELoss:
    def __init__(self, L0=0.05, H0=1200000.0, nu0=795774.715459):
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
            retain_graph=True,
            only_inputs=True
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
            retain_graph=True,
            only_inputs=True
        )[0]
        dHx_star_dy_star = grad_Hx_star[:, 1:2] * self.L0
        
        grad_Hy_star = torch.autograd.grad(
            outputs=H_y_star,
            inputs=xy,
            grad_outputs=torch.ones_like(H_y_star),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        dHy_star_dx_star = grad_Hy_star[:, 0:1] * self.L0
        
        residual_star = dHy_star_dx_star - dHx_star_dy_star - J_z_star
        return residual_star
'''
    with open(maxwell_loss_file, 'w', encoding='utf-8') as f:
        f.write(maxwell_loss_content)

    # 3. Cập nhật training_manager.py
    training_mgr_content = r'''import torch
import torch.optim as optim

class TrainingManager:
    def __init__(
        self, 
        model, 
        pde_evaluator, 
        learning_rate_adam=1e-3, 
        target_loss=0.0,
        lbfgs_learning_rate=0.8, 
        lbfgs_maximum_iterations=1000, 
        lbfgs_maximum_evaluations=1250,
        lbfgs_tolerance_gradient=1e-8,
        lbfgs_tolerance_change=1e-10,
        lbfgs_history_size=50,
        visualizer=None,
        visualizer_update_interval=100
    ):
        self.model = model
        self.pde_evaluator = pde_evaluator
        
        self.base_learning_rate_adam = learning_rate_adam
        self.target_loss = target_loss
        self.lbfgs_learning_rate = lbfgs_learning_rate
        self.lbfgs_maximum_iterations = lbfgs_maximum_iterations
        self.lbfgs_maximum_evaluations = lbfgs_maximum_evaluations
        self.lbfgs_tolerance_gradient = lbfgs_tolerance_gradient
        self.lbfgs_tolerance_change = lbfgs_tolerance_change
        self.lbfgs_history_size = lbfgs_history_size
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=self.lbfgs_maximum_iterations,
            max_eval=self.lbfgs_maximum_evaluations,
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )
        
        self.visualizer = visualizer
        self.visualizer_update_interval = visualizer_update_interval

    def compute_loss(self, domains_data):
        device = next(self.model.parameters()).device
        
        # Móc nối trực tiếp vào đồ thị autograd của tham số mô hình
        dummy_param = next(self.model.parameters())
        total_loss = (dummy_param * 0.0).sum()

        for key, data in domains_data.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                # Đảm bảo tensor tọa độ khả vi và độc lập
                pts = pts.detach().clone()
                pts.requires_grad_(True)
                    
                A_z_star = self.model.forward_network(key, pts)
                nu = domain.evaluate_reluctivity(pts).to(device)
                hx, hy = domain.evaluate_magnetization_vector(pts)
                hx, hy = hx.to(device), hy.to(device)
                jz = domain.evaluate_current_density(pts).to(device)
                
                residual_star = self.pde_evaluator.compute_residual(
                    xy=pts, A_z_star=A_z_star, nu=nu, J_z=jz, H_cx=hx, H_cy=hy
                )
                weighted_res = residual_star * domain.training_config.pde_weight
                pde_loss = torch.mean(weighted_res**2)
                total_loss = total_loss + pde_loss

            surf_pts = data["surface_points"].to(device)
            if surf_pts.shape[0] > 0 and key != 'background':
                surf_pts = surf_pts.detach().clone()
                surf_pts.requires_grad_(True)
                    
                Az_mat = self.model.forward_network(key, surf_pts)
                Az_bg = self.model.forward_network('background', surf_pts)
                
                continuity_loss = torch.mean((Az_mat - Az_bg)**2)
                total_loss = total_loss + domain.training_config.interface_weight * continuity_loss

        return total_loss

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        self.model.train()
        
        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_learning_rate_adam
            param_group['lr'] = self.base_learning_rate_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=epochs, eta_min=1e-6)
        stop_reason = f"Hoàn thành toàn bộ {epochs} epochs."
        
        for epoch in range(epochs):
            if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
                domains_data = resample_fn()

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            
            if torch.isnan(loss):
                print(f"Adam Epoch {epoch + 1}: CẢNH BÁO - Loss bị NaN. Bỏ qua step này.")
                continue
                
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            current_loss_value = loss.item()
            
            if self.target_loss > 0 and current_loss_value <= self.target_loss:
                stop_reason = f"Đạt ngưỡng target_loss ({self.target_loss}) tại epoch {epoch + 1}."
                print(f"Adam Epoch {epoch + 1}: {stop_reason} KẾT THÚC ADAM SỚM!")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Adam Epoch {epoch + 1}: Loss = {current_loss_value:.6e} | LR = {self.optimizer_adam.param_groups[0]['lr']:.3e}")
                
            if self.visualizer and (epoch + 1) % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
                
        print(f"[*] KẾT THÚC HUẤN LUYỆN ADAM. Lý do: {stop_reason}")

    def train_lbfgs(self, epochs, domains_data):
        self.model.train()
        lbfgs_counter = [0]
        early_stop_triggered = False 
        
        def closure():
            nonlocal early_stop_triggered
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            loss.backward(retain_graph=True)
            
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
                
            if self.visualizer and lbfgs_counter[0] % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(50000 + lbfgs_counter[0], loss.item(), domains_data)
                
            if self.target_loss > 0 and loss.item() <= self.target_loss:
                early_stop_triggered = True
            return loss
            
        for epoch in range(epochs):
            if early_stop_triggered:
                print(f"L-BFGS Epoch {epoch + 1}: Đạt ngưỡng target_loss. KẾT THÚC L-BFGS SỚM!")
                break
            self.optimizer_lbfgs.step(closure)
'''
    with open(training_mgr_file, 'w', encoding='utf-8') as f:
        f.write(training_mgr_content)

    print("[*] Đã sửa triệt để lỗi Autograd graph và đồng bộ hóa thành công.")

if __name__ == "__main__":
    fix_autograd_graph_issue()