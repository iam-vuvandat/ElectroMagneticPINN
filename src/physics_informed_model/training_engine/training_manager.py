import time
import torch
import torch.optim as optim

class TrainingManager:
    def __init__(self, electro_magnetic_pinn):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        
        t_cfg = self.electro_magnetic_pinn.training_config
        self.base_learning_rate_adam = t_cfg.learning_rate_adam
        self.target_loss = t_cfg.target_loss
        self.lbfgs_learning_rate = t_cfg.lbfgs_learning_rate
        self.lbfgs_maximum_iterations = t_cfg.lbfgs_maximum_iterations
        self.lbfgs_maximum_evaluations = t_cfg.lbfgs_maximum_evaluations
        self.lbfgs_tolerance_gradient = t_cfg.lbfgs_tolerance_gradient
        self.lbfgs_tolerance_change = t_cfg.lbfgs_tolerance_change
        self.lbfgs_history_size = t_cfg.lbfgs_history_size
        
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        self.optimizer_adam = optim.Adam(model.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=self.lbfgs_maximum_iterations,
            max_eval=self.lbfgs_maximum_evaluations,
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )
        self.last_loss_breakdown = {}
        self.completed_adam_epochs = 0

    @property
    def model(self):
        return self.electro_magnetic_pinn.pinn_architecture_instance

    @property
    def pde_evaluator(self):
        return self.electro_magnetic_pinn.maxwell_pde_loss_instance

    @property
    def interface_evaluator(self):
        return self.electro_magnetic_pinn.interface_loss_instance

    @property
    def visualizer(self):
        return self.electro_magnetic_pinn.visualizer_instance

    @property
    def visualizer_update_interval(self):
        return self.electro_magnetic_pinn.visualization_config.update_interval

    def compute_loss(self, domains_data):
        device = next(self.model.parameters()).device
        dummy_param = next(self.model.parameters())
        total_loss = (dummy_param * 0.0).sum()

        raw_pde_sum = 0.0
        weighted_pde_sum = 0.0
        az_if_sum = 0.0
        bn_if_sum = 0.0
        ht_if_sum = 0.0
        weighted_if_sum = 0.0

        for key, data in domains_data.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)
                    
                A_z_star = self.model.forward_network(key, pts)
                nu = domain.evaluate_reluctivity(pts).to(device)
                hx, hy = domain.evaluate_magnetization_vector(pts)
                hx, hy = hx.to(device), hy.to(device)
                jz = domain.evaluate_current_density(pts).to(device)
                
                residual_star = self.pde_evaluator.compute_residual(
                    xy=pts, A_z_star=A_z_star, nu=nu, J_z=jz, H_cx=hx, H_cy=hy
                )
                fading_w = self.pde_evaluator.compute_fading_weight(domain, pts)
                raw_pde = torch.mean((residual_star ** 2) * fading_w)
                pde_loss = float(domain.training_config.pde_weight) * raw_pde
                total_loss = total_loss + pde_loss

                raw_pde_sum += raw_pde.item()
                weighted_pde_sum += pde_loss.item()

            surf_pts = data["surface_points"].to(device)
            if surf_pts.shape[0] > 0 and key != 'background':
                if not surf_pts.requires_grad:
                    surf_pts = surf_pts.detach().clone().requires_grad_(True)
                    
                w_if_loss, val_l, bn_l, ht_l = self.interface_evaluator.compute_loss_components(
                    network_key=key,
                    domain=domain,
                    surf_pts=surf_pts,
                    background_key='background'
                )
                scaled_if_loss = float(domain.training_config.interface_weight) * w_if_loss
                total_loss = total_loss + scaled_if_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_if_loss.item()

        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        boundary_obj = sim_domain.boundary
        if getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor') == 'soft_loss':
            b_pts = boundary_obj.sample_boundary_points(device=device)
            if b_pts.shape[0] > 0:
                Az_b = self.model.forward_network('background', b_pts)
                target_b = float(boundary_obj.physics_config.potential_value) / max(sim_domain.A0, 1e-12)
                b_loss = float(boundary_obj.training_config.boundary_loss_weight) * torch.mean((Az_b - target_b) ** 2)
                total_loss = total_loss + b_loss

        self.last_loss_breakdown = {
            "raw_pde": raw_pde_sum,
            "weighted_pde": weighted_pde_sum,
            "interface_Az": az_if_sum,
            "interface_Bn": bn_if_sum,
            "interface_Ht": ht_if_sum,
            "weighted_interface": weighted_if_sum,
            "total": total_loss.item()
        }

        return total_loss

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        self.model.train()
        
        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_learning_rate_adam
            param_group['lr'] = self.base_learning_rate_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=max(epochs, 1), eta_min=1e-6)
        stop_reason = f"Hoàn thành toàn bộ {epochs} epochs."
        non_finite_streak = 0
        self.completed_adam_epochs = 0
        
        for epoch in range(epochs):
            if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
                domains_data = resample_fn()

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            
            if not torch.isfinite(loss):
                non_finite_streak += 1
                print(f"Adam Epoch {epoch + 1}: CẢNH BÁO - Loss không hữu hạn ({loss.item()}). Chuỗi lỗi: {non_finite_streak}")
                if non_finite_streak >= 5:
                    stop_reason = "Dừng sớm do Loss không hữu hạn (NaN/Inf) liên tiếp 5 lần."
                    break
                continue
            non_finite_streak = 0
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            self.completed_adam_epochs = epoch + 1
            current_loss_value = loss.item()
            
            if self.target_loss > 0 and current_loss_value <= self.target_loss:
                stop_reason = f"Đạt ngưỡng target_loss ({self.target_loss}) tại epoch {epoch + 1}."
                print(f"Adam Epoch {epoch + 1}: {stop_reason} KẾT THÚC ADAM SỚM!")
                break
            
            if (epoch + 1) == 1 or (epoch + 1) % 100 == 0:
                bd = self.last_loss_breakdown
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE(raw/w) = {bd.get('raw_pde', 0.0):.3e}/{bd.get('weighted_pde', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e} | "
                    f"LR = {self.optimizer_adam.param_groups[0]['lr']:.2e}"
                )
                
            if self.visualizer and (epoch + 1) % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
                
        print(f"[*] KẾT THÚC HUẤN LUYỆN ADAM. Lý do: {stop_reason}")
        return domains_data

    def train_lbfgs(self, max_iterations, domains_data):
        if max_iterations <= 0:
            return domains_data

        self.model.train()
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=max_iterations,
            max_eval=int(max_iterations * 1.25),
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )

        lbfgs_counter = [0]
        last_loss = [0.0]
        start_time = time.time()
        base_step = self.completed_adam_epochs
        stop_reason = f"Hoàn thành tối ưu hóa L-BFGS (tối đa {max_iterations} iterations)."
        
        def closure():
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            
            lbfgs_counter[0] += 1
            last_loss[0] = loss.item()
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 25 == 0:
                bd = self.last_loss_breakdown
                elapsed = time.time() - start_time
                print(
                    f"L-BFGS Closure {lbfgs_counter[0]:05d} ({elapsed:.1f}s) | Total = {loss.item():.4e} | "
                    f"PDE(w) = {bd.get('weighted_pde', 0.0):.3e} | IF(w) = {bd.get('weighted_interface', 0.0):.3e}"
                )
                
            if self.visualizer and lbfgs_counter[0] % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(base_step + lbfgs_counter[0], loss.item(), domains_data)
                
            return loss
            
        self.optimizer_lbfgs.step(closure)
        if self.visualizer and lbfgs_counter[0] > 0 and lbfgs_counter[0] % self.visualizer_update_interval != 0:
            self.visualizer.save_frame(base_step + lbfgs_counter[0], last_loss[0], domains_data)
        total_elapsed = time.time() - start_time
        print(f"[*] KẾT THÚC L-BFGS: {lbfgs_counter[0]} closure calls trong {total_elapsed:.2f}s. {stop_reason}")
        return domains_data
