import torch
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
        dummy_param = next(self.model.parameters())
        total_loss = (dummy_param * 0.0).sum()

        for key, data in domains_data.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.clone().detach().requires_grad_(True)
                    
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
                if not surf_pts.requires_grad:
                    surf_pts = surf_pts.clone().detach().requires_grad_(True)
                    
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
