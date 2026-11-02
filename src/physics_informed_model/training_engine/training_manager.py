import time
import torch
import torch.optim as optim

def resolve_domain_net_key(domain, arch_instance):
    """Xác định net_key khớp với khóa đã đăng ký trong XPINNArchitecture."""
    d_name = getattr(domain, 'domain_name', '')
    if d_name == 'background':
        return 'background'

    d_id = getattr(domain, 'domain_id', None)
    candidates = []
    if d_id is not None:
        candidates.append(f'domain_{d_id}')
    if d_name:
        candidates.append(d_name)

    if arch_instance is not None:
        if hasattr(arch_instance, 'nu_networks'):
            for k in candidates:
                if k in arch_instance.nu_networks:
                    return k
        if hasattr(arch_instance, 'networks'):
            for k in candidates:
                if k in arch_instance.networks:
                    return k

    return candidates[0] if candidates else 'background'


class TrainingManager:
    def __init__(self, electro_magnetic_pinn):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        
        t_cfg = self.electro_magnetic_pinn.training_config
        
        self.base_learning_rate_adam = getattr(t_cfg, 'adam_learning_rate', getattr(t_cfg, 'learning_rate_adam', 1e-3))
        self.target_loss_threshold = getattr(t_cfg, 'target_loss_threshold', getattr(t_cfg, 'target_loss', 1e-5))
        self.lbfgs_learning_rate = getattr(t_cfg, 'lbfgs_learning_rate', 0.5)
        self.lbfgs_maximum_iterations = getattr(t_cfg, 'number_of_lbfgs_iterations', getattr(t_cfg, 'lbfgs_maximum_iterations', 1500))
        self.lbfgs_maximum_evaluations = getattr(t_cfg, 'lbfgs_maximum_evaluations', 2500)
        self.lbfgs_tolerance_gradient = getattr(t_cfg, 'lbfgs_tolerance_gradient', 1e-12)
        self.lbfgs_tolerance_change = getattr(t_cfg, 'lbfgs_tolerance_change', 1e-12)
        self.lbfgs_history_size = getattr(t_cfg, 'lbfgs_history_size', 50)
        
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        
        self.optimizer_adam = optim.Adam(model.networks.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            model.networks.parameters(),
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

    def pretrain_nu_networks(self, domains_data, epochs=None, lr=None):
        if not hasattr(self.model, 'nu_networks') or len(self.model.nu_networks) == 0:
            return

        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)
        epochs = epochs if epochs is not None else getattr(t_cfg, 'pretrain_nu_epochs', 300)
        lr = lr if lr is not None else getattr(t_cfg, 'pretrain_nu_lr', 1e-2)

        print(f"\n[*] Pre-training Mạng phụ Net_nu về hằng số ban đầu ({epochs} epochs, lr={lr})...")
        device = next(self.model.parameters()).device
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        
        nu_params = []
        for net in self.model.nu_networks.values():
            nu_params.extend(net.parameters())
            
        opt_nu = optim.Adam(nu_params, lr=lr)
        domains_dict = domains_data.get("domains", domains_data)
        
        for ep in range(epochs):
            opt_nu.zero_grad()
            loss_pre = torch.tensor(0.0, device=device)
            
            for key, data in domains_dict.items():
                domain = data["domain_object"]
                net_key = resolve_domain_net_key(domain, self.model)
                if net_key in self.model.nu_networks:
                    pts = data["interior_points"].to(device)
                    
                    if pts.shape[0] > 0:
                        nu_pred_star = self.model.forward_nu_network(net_key, pts)
                        nu_target_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                        nu_target_star = nu_target_phys / sim_domain.nu0
                        
                        loss_domain = torch.mean((nu_pred_star - nu_target_star) ** 2)
                        loss_pre = loss_pre + loss_domain
                        
            if loss_pre.requires_grad:
                loss_pre.backward()
                opt_nu.step()

    def fit_nu_network_step(self, domains_data, alpha_damping=None, epochs=None, lr=None):
        if not hasattr(self.model, 'nu_networks') or len(self.model.nu_networks) == 0:
            return 0.0

        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)
        alpha_damping = alpha_damping if alpha_damping is not None else getattr(t_cfg, 'fpi_alpha_damping', 0.25)
        epochs = epochs if epochs is not None else getattr(t_cfg, 'fit_nu_epochs', 250)
        lr = lr if lr is not None else getattr(t_cfg, 'fit_nu_lr', 3e-3)

        device = next(self.model.parameters()).device
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        domains_dict = domains_data.get("domains", domains_data)
        
        targets_dict = {}
        max_nu_diff = 0.0

        for key, data in domains_dict.items():
            domain = data["domain_object"]
            net_key = resolve_domain_net_key(domain, self.model)
            if net_key in self.model.nu_networks:
                if not getattr(domain.physics_config, 'is_nonlinear', False):
                    continue

                pts = data["interior_points"].to(device)
                if pts.shape[0] > 0:
                    pts_grad = pts.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        Az_star = self.model.forward_network(net_key, pts_grad)
                        grad_Az = torch.autograd.grad(
                            outputs=Az_star, inputs=pts_grad,
                            grad_outputs=torch.ones_like(Az_star), create_graph=False
                        )[0]
                        Bx_star = grad_Az[:, 1:2] * sim_domain.L0
                        By_star = -grad_Az[:, 0:1] * sim_domain.L0
                        B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * sim_domain.B0
                        
                    with torch.no_grad():
                        nu_BH_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys.detach(), is_nonlinear=True)
                        nu_BH_star = (nu_BH_phys / sim_domain.nu0).detach()
                        
                        nu_old_star = self.model.forward_nu_network(net_key, pts).detach()
                        nu_target_star = (1.0 - alpha_damping) * nu_old_star + alpha_damping * nu_BH_star
                        
                        diff_val = torch.max(torch.abs(nu_target_star - nu_old_star)).item()
                        max_nu_diff = max(max_nu_diff, diff_val)
                        targets_dict[net_key] = (pts, nu_target_star)

        if not targets_dict:
            return 0.0

        nu_params = []
        for k in targets_dict.keys():
            nu_params.extend(self.model.nu_networks[k].parameters())
        opt_nu = optim.Adam(nu_params, lr=lr)

        for _ in range(epochs):
            opt_nu.zero_grad()
            total_fit_loss = torch.tensor(0.0, device=device)
            for k, (pts, nu_target_star) in targets_dict.items():
                nu_pred_star = self.model.forward_nu_network(k, pts)
                total_fit_loss = total_fit_loss + torch.mean((nu_pred_star - nu_target_star) ** 2)
            total_fit_loss.backward()
            opt_nu.step()

        return max_nu_diff

    def compute_loss(self, domains_data):
        device = next(self.model.parameters()).device
        dummy_param = next(self.model.parameters())
        total_loss = (dummy_param * 0.0).sum()

        pde_loss_sum = 0.0
        az_if_sum = 0.0
        bn_if_sum = 0.0
        ht_if_sum = 0.0
        weighted_if_sum = 0.0

        domains_dict = domains_data.get("domains", domains_data)
        shared_dict = domains_data.get("shared_interfaces", {})
        bg_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance.background_domain

        for key, data in domains_dict.items():
            domain = data["domain_object"]
            net_key = resolve_domain_net_key(domain, self.model)
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)
                    
                A_z_star = self.model.forward_network(net_key, pts)
                
                # CẮT ĐỒ THỊ ĐẠO HÀM DÙNG DETACH() ĐỂ TRÁNH RÒ RỈ GRADIENT VÀO ADAM
                if hasattr(self.model, 'nu_networks') and net_key in self.model.nu_networks:
                    nu_star_pred = self.model.forward_nu_network(net_key, pts)
                    nu_star_net = nu_star_pred.detach() if nu_star_pred is not None else None
                else:
                    nu_star_net = None

                nu_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                hx, hy = domain.evaluate_magnetization_vector(pts)
                hx, hy = hx.to(device), hy.to(device)
                jz = domain.evaluate_current_density(pts).to(device)
                
                res_pde = self.pde_evaluator.compute_linear_pde_residual(
                    xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_star_net=nu_star_net
                )
                fading_w = self.pde_evaluator.compute_fading_weight(domain, pts)
                
                loss_pde_domain = torch.mean((res_pde ** 2) * fading_w)
                w_pde = float(domain.training_config.pde_weight)

                scaled_pde_loss = w_pde * loss_pde_domain
                total_loss = total_loss + scaled_pde_loss
                pde_loss_sum += scaled_pde_loss.item()

            outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
            if outer_pts.shape[0] > 0 and net_key != 'background':
                if not outer_pts.requires_grad:
                    outer_pts = outer_pts.detach().clone().requires_grad_(True)
                    
                w_if_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
                )
                scaled_if_loss = w_if_loss
                total_loss = total_loss + scaled_if_loss

                az_if_sum += val_l.item()
                bn_if_sum += bn_l.item()
                ht_if_sum += ht_l.item()
                weighted_if_sum += scaled_if_loss.item()

        for pair_key, pair_data in shared_dict.items():
            pts = pair_data["interface_points"].to(device)
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)

                w_shared_loss, val_l, bn_l, ht_l = self.interface_evaluator._compute_pair_components(
                    key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                    key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts
                )
                scaled_shared_loss = w_shared_loss
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
        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)

        eta_min = getattr(t_cfg, 'adam_eta_min', 1e-6)
        max_clip_norm = getattr(t_cfg, 'grad_clip_max_norm', 1.0)
        max_non_finite_streak = getattr(t_cfg, 'max_non_finite_streak', 5)
        log_interval = getattr(t_cfg, 'adam_log_interval', 100)

        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_learning_rate_adam
            param_group['lr'] = self.base_learning_rate_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=max(epochs, 1), eta_min=eta_min)
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
                if non_finite_streak >= max_non_finite_streak:
                    stop_reason = f"Dừng sớm do Loss không hữu hạn (NaN/Inf) liên tiếp {max_non_finite_streak} lần."
                    break
                continue
            non_finite_streak = 0
                
            loss.backward()
            if max_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.networks.parameters(), max_norm=max_clip_norm)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            self.completed_adam_epochs = epoch + 1
            current_loss_value = loss.item()
            
            if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0:
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
        t_cfg = getattr(self.electro_magnetic_pinn, 'training_config', None)
        log_interval = getattr(t_cfg, 'lbfgs_log_interval', 25)

        self.optimizer_lbfgs = optim.LBFGS(
            self.model.networks.parameters(),
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
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % log_interval == 0:
                bd = self.last_loss_breakdown
                print(
                    f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {loss.item():.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e}"
                )
            return loss
            
        self.optimizer_lbfgs.step(closure)
        return domains_data
