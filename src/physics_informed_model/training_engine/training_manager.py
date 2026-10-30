import time
import torch
import torch.optim as optim

class TrainingManager:
    def __init__(self, electro_magnetic_pinn):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        
        t_cfg = self.electro_magnetic_pinn.training_config
        
        self.base_learning_rate_adam = getattr(t_cfg, 'adam_learning_rate', getattr(t_cfg, 'learning_rate_adam', 1e-3))
        self.target_loss_threshold = getattr(t_cfg, 'target_loss_threshold', getattr(t_cfg, 'target_loss', 1e-5))
        self.lbfgs_learning_rate = getattr(t_cfg, 'lbfgs_learning_rate', 0.8)
        self.lbfgs_maximum_iterations = getattr(t_cfg, 'number_of_lbfgs_iterations', getattr(t_cfg, 'lbfgs_maximum_iterations', 800))
        self.lbfgs_maximum_evaluations = getattr(t_cfg, 'lbfgs_maximum_evaluations', 1000)
        self.lbfgs_tolerance_gradient = getattr(t_cfg, 'lbfgs_tolerance_gradient', 1e-8)
        self.lbfgs_tolerance_change = getattr(t_cfg, 'lbfgs_tolerance_change', 1e-10)
        self.lbfgs_history_size = getattr(t_cfg, 'lbfgs_history_size', 50)
        
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

        pde_loss_sum = 0.0
        az_if_sum = 0.0
        bn_if_sum = 0.0
        ht_if_sum = 0.0
        weighted_if_sum = 0.0

        if "domains" in domains_data:
            domains_dict = domains_data["domains"]
            shared_dict = domains_data.get("shared_interfaces", {})
        else:
            domains_dict = domains_data
            shared_dict = {}

        bg_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance.background_domain

        # 1. PDE Loss
        for key, data in domains_dict.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)
                    
                A_z_star = self.model.forward_network(key, pts)
                nu_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                hx, hy = domain.evaluate_magnetization_vector(pts)
                hx, hy = hx.to(device), hy.to(device)
                jz = domain.evaluate_current_density(pts).to(device)
                
                res_pde = self.pde_evaluator.compute_linear_pde_residual(
                    xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy
                )
                fading_w = self.pde_evaluator.compute_fading_weight(domain, pts)
                
                loss_pde_domain = torch.mean((res_pde ** 2) * fading_w)
                w_pde = float(domain.training_config.pde_weight)

                scaled_pde_loss = w_pde * loss_pde_domain
                total_loss = total_loss + scaled_pde_loss
                pde_loss_sum += scaled_pde_loss.item()

            outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
            if outer_pts.shape[0] > 0 and key != 'background':
                if not outer_pts.requires_grad:
                    outer_pts = outer_pts.detach().clone().requires_grad_(True)
                    
                w_if_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=key,
                    domain_a=domain,
                    key_b='background',
                    domain_b=bg_domain,
                    pts=outer_pts
                )
                scaled_if_loss = float(domain.training_config.interface_weight) * w_if_loss
                total_loss = total_loss + scaled_if_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_if_loss.item()

        # 2. Shared Interface Loss
        for pair_key, pair_data in shared_dict.items():
            pts = pair_data["interface_points"].to(device)
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)

                w_shared_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=pair_data["key_a"],
                    domain_a=pair_data["domain_a"],
                    key_b=pair_data["key_b"],
                    domain_b=pair_data["domain_b"],
                    pts=pts
                )
                w_weight = 0.5 * (float(pair_data["domain_a"].training_config.interface_weight) + 
                                  float(pair_data["domain_b"].training_config.interface_weight))
                scaled_shared_loss = w_weight * w_shared_loss
                total_loss = total_loss + scaled_shared_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_shared_loss.item()

        self.last_loss_breakdown = {
            "pde_loss": pde_loss_sum,
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
            
            if (epoch + 1) == 1 or (epoch + 1) % 100 == 0:
                bd = self.last_loss_breakdown
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
                )
                
            if self.visualizer and (epoch + 1) % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
                
        print(f"[*] KẾT THÚC ADAM. Lý do: {stop_reason}")
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
        
        def closure():
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 25 == 0:
                bd = self.last_loss_breakdown
                print(
                    f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {loss.item():.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e}"
                )
            return loss
            
        self.optimizer_lbfgs.step(closure)
        return domains_data
