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

    def pretrain_nu_networks(self, domains_data, epochs=300):
        """
        Huấn luyện ấm Mạng phụ Net_nu để khớp với độ từ kháng ban đầu nu_init*.
        """
        if not hasattr(self.model, 'nu_networks') or len(self.model.nu_networks) == 0:
            return

        print(f"\n[*] Bắt đầu Pre-training Mạng phụ $Net_{{\\nu}}$ ({epochs} epochs)...")
        device = next(self.model.parameters()).device
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        
        nu_params = []
        for net in self.model.nu_networks.values():
            nu_params.extend(net.parameters())
            
        opt_nu = optim.Adam(nu_params, lr=1e-2)
        domains_dict = domains_data.get("domains", domains_data)
        
        for ep in range(epochs):
            opt_nu.zero_grad()
            loss_pre = 0.0
            
            for key, data in domains_dict.items():
                if key in self.model.nu_networks:
                    domain = data["domain_object"]
                    pts = data["interior_points"].to(device)
                    
                    if pts.shape[0] > 0:
                        nu_pred_star = self.model.forward_nu_network(key, pts)
                        nu_target_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                        nu_target_star = nu_target_phys / sim_domain.nu0
                        
                        loss_domain = torch.mean((nu_pred_star - nu_target_star) ** 2)
                        loss_pre = loss_pre + loss_domain
                        
            if loss_pre > 0:
                loss_pre.backward()
                opt_nu.step()
                
            if (ep + 1) == 1 or (ep + 1) % 100 == 0:
                print(f"Pre-train Net_nu Epoch {ep + 1:03d} | Loss_nu = {loss_pre.item():.4e}")
                
        print("[✔] Hoàn thành Pre-train Mạng phụ $Net_{\\nu}$!")

    def train_fpi(self, max_fpi_iterations, inner_adam_epochs, domains_data, relaxation_alpha=0.5):
        """
        Vòng lặp ngoài Fixed-Point Iteration (FPI/Picard) kết hợp làm mềm relaxation.
        """
        print(f"\n[*] BẮT ĐẦU VÒNG LẶP FIXED-POINT ITERATION (FPI) [Tối đa {max_fpi_iterations} vòng ngoài]...")
        device = next(self.model.parameters()).device
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance

        for fpi_iter in range(max_fpi_iterations):
            print(f"\n=================== [FPI OUTER LOOP {fpi_iter + 1}/{max_fpi_iterations}] ===================")
            
            # 1. Huấn luyện vòng trong XPINN cho Net_Az với nu đóng băng
            domains_data = self.train_adam(
                epochs=inner_adam_epochs, 
                domains_data=domains_data
            )

            # 2. Đánh giá trường B từ nghiệm Az hiện tại và cập nhật nu_target từ B-H lookup
            domains_dict = domains_data.get("domains", domains_data)
            max_nu_diff = 0.0

            for key, data in domains_dict.items():
                if key in self.model.nu_networks:
                    domain = data["domain_object"]
                    pts = data["interior_points"].to(device)
                    
                    if pts.shape[0] > 0:
                        # Tính B_mag từ Az hiện tại
                        Az_star = self.model.forward_network(key, pts)
                        grad_A = torch.autograd.grad(outputs=Az_star, inputs=pts, grad_outputs=torch.ones_like(Az_star), create_graph=False)[0]
                        Bx_star = grad_A[:, 1:2] * sim_domain.L0
                        By_star = -grad_A[:, 0:1] * sim_domain.L0
                        B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * sim_domain.B0

                        # Tra cứu nu_target từ B-H curve
                        nu_target_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys, is_nonlinear=True).to(device)
                        nu_target_star = nu_target_phys / sim_domain.nu0

                        # Lấy nu_current từ Net_nu
                        nu_current_star = self.model.forward_nu_network(key, pts).detach()

                        # Áp dụng Thư giãn (Relaxation / Damping)
                        nu_relaxed_star = (1.0 - relaxation_alpha) * nu_current_star + relaxation_alpha * nu_target_star

                        # Refit Mạng phụ Net_nu theo nu_relaxed_star
                        nu_net = self.model.nu_networks[key]
                        opt_sub_nu = optim.Adam(nu_net.parameters(), lr=1e-3)
                        for _ in range(80):
                            opt_sub_nu.zero_grad()
                            nu_pred = self.model.forward_nu_network(key, pts)
                            loss_fit = torch.mean((nu_pred - nu_relaxed_star) ** 2)
                            loss_fit.backward()
                            opt_sub_nu.step()

                        diff = torch.max(torch.abs(nu_relaxed_star - nu_current_star)).item()
                        max_nu_diff = max(max_nu_diff, diff)

            print(f"[*] FPI Loop {fpi_iter + 1} Hoàn thành | Độ thay đổi nu* lớn nhất: {max_nu_diff:.4e}")
            if max_nu_diff < 1e-4:
                print(f"[✔] FPI Hội tụ sớm tại vòng {fpi_iter + 1}!")
                break

        return domains_data

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

        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        bg_domain = sim_domain.background_domain

        # 1. PDE Loss
        for key, data in domains_dict.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)
                    
                A_z_star = self.model.forward_network(key, pts)
                
                # Tra cứu nu* trực tiếp từ Net_nu nếu domain có Net_nu
                if hasattr(self.model, 'nu_networks') and key in self.model.nu_networks:
                    nu_star_net = self.model.forward_nu_network(key, pts)
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
        stop_reason = f"Hoàn thành {epochs} epochs."
        
        for epoch in range(epochs):
            if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
                domains_data = resample_fn()

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(domains_data)
            
            if not torch.isfinite(loss):
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            self.completed_adam_epochs = epoch + 1
            if (epoch + 1) == 1 or (epoch + 1) % 100 == 0:
                bd = self.last_loss_breakdown
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {loss.item():.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e}"
                )
                
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
