import os

def apply_permeability_and_project_fixes():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    forward_net_file = os.path.join(project_root, 'architectures', 'forward_network.py')
    xpinn_arch_file = os.path.join(project_root, 'architectures', 'xpinn_architecture.py')
    pinn_core_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    training_mgr_file = os.path.join(project_root, 'training_engine', 'training_manager.py')
    visualizer_file = os.path.join(project_root, 'utils', 'training_visualizer.py')

    # =========================================================================
    # 1. FIX: architectures/forward_network.py (Tra cứu key linh hoạt)
    # =========================================================================
    forward_net_code = '''import torch
import torch.nn as nn
import numpy as np

def forward_network(self, network_key, xy):
    actual_key = network_key
    if actual_key not in self.networks:
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        for domain in sim_domain.domains_list:
            d_name = getattr(domain, 'domain_name', '')
            d_id = getattr(domain, 'domain_id', None)
            if d_name == network_key or f"domain_{d_id}" == network_key:
                if f"domain_{d_id}" in self.networks:
                    actual_key = f"domain_{d_id}"
                    break
                elif d_name in self.networks:
                    actual_key = d_name
                    break

    if actual_key not in self.networks:
        raise ValueError(f"Lỗi: Không tìm thấy mạng con '{network_key}' trong XPINNArchitecture.")
        
    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == actual_key or f"domain_{domain.domain_id}" == actual_key:
            target_domain = domain
            break
            
    if target_domain is None:
        raise ValueError(f"Lỗi: Không tìm thấy domain tương ứng với network_key '{actual_key}'.")

    config = target_domain.network_config
    
    if config.fourier_features:
        B_matrix = getattr(self, f'B_matrix_{actual_key}', None)
        if B_matrix is not None:
            if B_matrix.device != xy.device:
                B_matrix = B_matrix.to(xy.device)
            projected = (2.0 * np.pi * xy_normalized) @ B_matrix.T
            x_input = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        else:
            x_input = xy_normalized
    else:
        x_input = xy_normalized
        
    raw_output = self.networks[actual_key](x_input)
    
    if actual_key == 'background' or getattr(target_domain, 'domain_name', '') == 'background':
        boundary_obj = sim_domain.boundary
        enforce_method = getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor')
        pot_val_star = float(getattr(boundary_obj.physics_config, 'potential_value', 0.0)) / max(sim_domain.A0, 1e-12)
        if enforce_method == 'hard_factor':
            A_z_star = raw_output * self.boundary_factor(xy) + pot_val_star
        else:
            A_z_star = raw_output
    else:
        A_z_star = raw_output
        
    return A_z_star


def forward_nu_network(self, network_key, xy):
    """
    Tra cứu độ từ kháng nu* = nu / nu0 từ Mạng phụ Net_nu.
    Hỗ trợ đối chiếu tự động giữa tên domain và domain_id.
    """
    actual_key = network_key
    if actual_key not in self.nu_networks:
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        for domain in sim_domain.domains_list:
            d_name = getattr(domain, 'domain_name', '')
            d_id = getattr(domain, 'domain_id', None)
            if d_name == network_key or f"domain_{d_id}" == network_key:
                if f"domain_{d_id}" in self.nu_networks:
                    actual_key = f"domain_{d_id}"
                    break
                elif d_name in self.nu_networks:
                    actual_key = d_name
                    break

    if actual_key not in self.nu_networks:
        return None

    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == actual_key or f"domain_{domain.domain_id}" == actual_key:
            target_domain = domain
            break

    nu_config = getattr(target_domain, 'nu_network_config', None) if target_domain else None

    if nu_config and getattr(nu_config, 'fourier_features', False):
        B_matrix_nu = getattr(self, f'B_matrix_nu_{actual_key}', None)
        if B_matrix_nu is not None:
            if B_matrix_nu.device != xy.device:
                B_matrix_nu = B_matrix_nu.to(xy.device)
            projected = (2.0 * np.pi * xy_normalized) @ B_matrix_nu.T
            x_input = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        else:
            x_input = xy_normalized
    else:
        x_input = xy_normalized

    raw_nu = self.nu_networks[actual_key](x_input)
    # Softplus + 1e-6 đảm bảo nu* luôn dương
    nu_star = torch.nn.functional.softplus(raw_nu) + 1e-6
    return nu_star
'''

    # =========================================================================
    # 2. FIX: architectures/xpinn_architecture.py (Khởi tạo np.expm1 chính xác)
    # =========================================================================
    xpinn_arch_code = '''import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace

from .build_single_network import _build_single_network
from .initialize_weights import _initialize_weights
from .boundary_factor import boundary_factor
from .forward_network import forward_network, forward_nu_network
from .forward import forward

class XPINNArchitecture(nn.Module):
    _build_single_network = _build_single_network
    _initialize_weights = _initialize_weights
    boundary_factor = boundary_factor
    forward_network = forward_network
    forward_nu_network = forward_nu_network
    forward = forward

    def __init__(self, electro_magnetic_pinn, input_dim=2, output_dim=1, domain_scale=None):
        super().__init__()
        
        self.electro_magnetic_pinn = electro_magnetic_pinn
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        self.domain_scale = domain_scale if domain_scale is not None else sim_domain.L0
        
        self.networks = nn.ModuleDict()
        self.nu_networks = nn.ModuleDict()
        
        self.domain_count = sim_domain.get_domain_count()
        domains_list = sim_domain.domains_list
        
        for domain in domains_list:
            if getattr(domain, 'domain_name', '') == 'background':
                net_key = 'background'
            else:
                net_key = f'domain_{domain.domain_id}'
                
            # 1. Dựng Mạng chính (Net_Az)
            config = domain.network_config
            if config.fourier_features:
                actual_input_dim = config.fourier_mapping_size * 2
                B_mat = torch.randn((config.fourier_mapping_size, self.input_dim)) * config.fourier_sigma
                self.register_buffer(f'B_matrix_{net_key}', B_mat)
            else:
                actual_input_dim = self.input_dim
                
            sub_net = self._build_single_network(actual_input_dim, config)
            self.networks[net_key] = sub_net

            # 2. Dựng Mạng phụ (Net_nu) cho các miền vật liệu
            if net_key != 'background':
                nu_config = getattr(domain, 'nu_network_config', SimpleNamespace(
                    hidden_layers=2, hidden_neurons=32, activation_function=nn.SiLU(), fourier_features=False
                ))
                
                if getattr(nu_config, 'fourier_features', False):
                    actual_nu_input_dim = nu_config.fourier_mapping_size * 2
                    B_mat_nu = torch.randn((nu_config.fourier_mapping_size, self.input_dim)) * nu_config.fourier_sigma
                    self.register_buffer(f'B_matrix_nu_{net_key}', B_mat_nu)
                else:
                    actual_nu_input_dim = self.input_dim

                nu_net = self._build_single_network(actual_nu_input_dim, nu_config)
                self.nu_networks[net_key] = nu_net
                
                self._initialize_nu_network_constant(domain, net_key)

        self._initialize_weights()

    def _initialize_nu_network_constant(self, domain, net_key):
        """Khởi tạo Net_nu vừa bám giá trị ban đầu vừa giữ chính xác số học cho độ từ thẩm lớn."""
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        dummy_pts = torch.zeros((1, 2))
        nu_target_phys = domain.evaluate_reluctivity(dummy_pts, is_nonlinear=False).item()
        nu_target_star = max(nu_target_phys / sim_domain.nu0, 1e-6)
        
        if nu_target_star < 20.0:
            # Sử dụng np.expm1 để tránh lỗi mất chính xác khi nu_target_star rất nhỏ
            target_raw = np.log(max(np.expm1(nu_target_star), 1e-12))
        else:
            target_raw = nu_target_star

        net = self.nu_networks[net_key]
        for module in net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)
                
        last_linear = [m for m in net if isinstance(m, nn.Linear)][-1]
        nn.init.constant_(last_linear.bias, target_raw)
'''

    # =========================================================================
    # 3. FIX: training_engine/training_manager.py (Ngắt autograd graph leakage)
    # =========================================================================
    training_mgr_code = '''import time
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

        print(f"\\n[*] Pre-training Mạng phụ Net_nu về hằng số ban đầu ({epochs} epochs, lr={lr})...")
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
'''

    # =========================================================================
    # 4. FIX: electro_magnetic_pinn.py (plot_fields ép kiểu tensor an toàn)
    # =========================================================================
    pinn_core_code = '''import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from types import SimpleNamespace
from architectures.xpinn_architecture import XPINNArchitecture
from training_engine.training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from physics_domain.physical_equations.interface_loss import InterfaceLoss
from simulation_domain_engine.simulation_domain import SimulationDomain
from physics_domain.collocation_sampler import CollocationSampler
from utils.training_visualizer import TrainingVisualizer
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

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


class ElectroMagneticPINN:
    def __init__(self):
        self.sampler_config = SimpleNamespace(
            dynamic_resampling=True,
            resample_interval=100,
            resample_points_before_lbfgs_stage=True
        )
        
        self.training_config = SimpleNamespace(
            enable_linear_warmup_stage=True,
            enable_nonlinear_adaptation_stage=True,
            enable_lbfgs_refinement_stage=True,
            linear_warmup_epochs_ratio=0.3,

            adam_learning_rate=1e-3,
            number_of_adam_epochs=2500,
            target_loss_threshold=1e-5,

            fpi_outer_iterations=10,
            fpi_alpha_damping=1.0,
            fpi_tolerance=1e-5,

            pretrain_nu_epochs=300,
            pretrain_nu_lr=1e-2,
            fit_nu_epochs=250,
            fit_nu_lr=3e-3,

            grad_clip_max_norm=1.0,
            adam_eta_min=1e-6,
            max_non_finite_streak=5,
            adam_log_interval=100,
            lbfgs_log_interval=25,

            lbfgs_learning_rate=0.5,
            number_of_lbfgs_iterations=1500,
            lbfgs_maximum_evaluations=2500,
            lbfgs_tolerance_gradient=1e-12,
            lbfgs_tolerance_change=1e-12,
            lbfgs_history_size=50
        )
        
        self.visualization_config = SimpleNamespace(
            active=False,
            update_interval=200,
            output_directory="animation_frames",
            resolution=80,
            generate_gif_animation=True,
            gif_filename="training_process.gif",
            generate_mp4_video=True,
            mp4_filename="training_process.mp4",
            animation_fps=5,
            gif_fps=5,
            mp4_codec="libx264",
            mp4_quality=8,

            contour_levels=60,

            res_scale="log",          
            res_vmin=1e-4,            
            res_vmax=1e1,             
            res_cmap="Reds",

            mur_scale="linear",       
            mur_vmin=0.0,             
            mur_vmax=3000.0,          
            mur_cmap="turbo",         

            az_scale="linear",        
            az_vmin=None,             
            az_vmax=None,             
            az_cmap="jet",

            b_mag_scale="linear",     
            b_mag_vmin=0.0,           
            b_mag_vmax=1.8,           
            b_mag_cmap="jet",

            by_scale="linear",        
            by_vmin=None,             
            by_vmax=None,             
            by_cmap="coolwarm"
        )
        
        self.simulation_domain_engine_instance = SimulationDomain(electro_magnetic_pinn=self)
        self.pinn_architecture_instance = None
        self.training_manager_instance = None
        self.maxwell_pde_loss_instance = None
        self.interface_loss_instance = None
        self.collocation_sampler_instance = None
        self.visualizer_instance = None

    @property
    def physics_config(self):
        return self.simulation_domain_engine_instance.physics_config

    @property
    def L0(self):
        return self.simulation_domain_engine_instance.L0

    @property
    def H0(self):
        return self.simulation_domain_engine_instance.H0

    @property
    def nu0(self):
        return self.simulation_domain_engine_instance.nu0

    @property
    def A0(self):
        return self.simulation_domain_engine_instance.A0

    def update_electromagnetic_pinn(self):
        self._build_system()

    def _build_system(self):
        self.simulation_domain_engine_instance.validate_non_overlapping_domains()
        self.computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Hệ thống vật lý được khởi tạo trên: {self.computation_device}")
        
        self.collocation_sampler_instance = CollocationSampler(
            electro_magnetic_pinn=self,
            device=self.computation_device
        )
        
        self.pinn_architecture_instance = XPINNArchitecture(
            electro_magnetic_pinn=self
        ).to(self.computation_device)
        
        self.maxwell_pde_loss_instance = MaxwellPDELoss(electro_magnetic_pinn=self)
        self.interface_loss_instance = InterfaceLoss(electro_magnetic_pinn=self)
        
        self.visualizer_instance = None
        if self.visualization_config.active:
            self.visualizer_instance = TrainingVisualizer(electro_magnetic_pinn=self)
            
        self.training_manager_instance = TrainingManager(electro_magnetic_pinn=self)

        domains_data = self.collocation_sampler_instance.generate_all_domain_points()
        pre_ep = getattr(self.training_config, 'pretrain_nu_epochs', 300)
        pre_lr = getattr(self.training_config, 'pretrain_nu_lr', 1e-2)
        self.training_manager_instance.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)

    def evaluate_fields(self, points_tensor, return_numpy=False):
        pinn_arch = self.pinn_architecture_instance
        sim_domain = self.simulation_domain_engine_instance
        device = points_tensor.device

        phys_props = sim_domain.evaluate_global_physical_properties(points_tensor)
        mat_id = phys_props["material_classification"]

        A_z_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
        B_x_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
        B_y_phys = torch.zeros((points_tensor.shape[0], 1), device=device)

        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, pinn_arch)
            mask_bool = (mat_id == float(d_id))
            
            if mask_bool.any():
                Az_star = pinn_arch.forward_network(net_key, points_tensor)
                grad_star = torch.autograd.grad(
                    outputs=Az_star, inputs=points_tensor,
                    grad_outputs=torch.ones_like(Az_star), create_graph=False
                )[0]

                Az_phys = Az_star * self.A0
                Bx_phys = grad_star[:, 1:2] * self.A0
                By_phys = -grad_star[:, 0:1] * self.A0

                A_z_phys = torch.where(mask_bool, Az_phys, A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_phys, B_x_phys)
                B_y_phys = torch.where(mask_bool, By_phys, B_y_phys)

        if return_numpy:
            return (
                A_z_phys.detach().cpu().numpy(),
                B_x_phys.detach().cpu().numpy(),
                B_y_phys.detach().cpu().numpy()
            )

        return A_z_phys.detach(), B_x_phys.detach(), B_y_phys.detach()

    def plot_fields(self, resolution=180, save_path=None, show=False, figsize=(12, 10)):
        sim_domain = self.simulation_domain_engine_instance
        vis_cfg = self.visualization_config
        pinn_arch = self.pinn_architecture_instance

        xb, yb = sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple
        x_coords = np.linspace(xb[0], xb[1], resolution)
        y_coords = np.linspace(yb[0], yb[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        
        device = self.computation_device
        eval_pts = torch.tensor(
            np.column_stack((X_grid.ravel(), Y_grid.ravel())),
            dtype=torch.float32, device=device
        ).requires_grad_(True)
        
        A_z_np, B_x_np, B_y_np = self.evaluate_fields(eval_pts, return_numpy=True)

        A_z_grid = A_z_np.reshape(resolution, resolution)
        B_x_grid = B_x_np.reshape(resolution, resolution)
        B_y_grid = B_y_np.reshape(resolution, resolution)
        B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)

        props = sim_domain.evaluate_global_physical_properties(eval_pts)
        mat_ids = props["material_classification"].squeeze()
        
        mu_r_tensor = torch.ones_like(mat_ids, device=device)
        B_mag_pts = torch.tensor(B_mag_grid.ravel(), dtype=torch.float32, device=device).unsqueeze(1)

        for domain in sim_domain.domains_list:
            if getattr(domain, 'domain_name', '') != 'background':
                d_id = domain.domain_id
                net_key = resolve_domain_net_key(domain, pinn_arch)
                mask = (mat_ids == float(d_id))
                if mask.any():
                    if getattr(domain.physics_config, 'is_nonlinear', False):
                        if hasattr(pinn_arch, 'nu_networks') and net_key in pinn_arch.nu_networks:
                            with torch.no_grad():
                                nu_star_pred = pinn_arch.forward_nu_network(net_key, eval_pts[mask])
                                mu_r_vals = (1.0 / torch.clamp(nu_star_pred, min=1e-6)).view(-1)
                                mu_r_tensor[mask] = mu_r_vals
                        else:
                            nu_phys = domain.evaluate_reluctivity(eval_pts[mask], B_magnitude_tensor=B_mag_pts[mask], is_nonlinear=True)
                            mu_r_vals = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys, min=1e-6)).view(-1)
                            mu_r_tensor[mask] = mu_r_vals
                    else:
                        mu_val = float(getattr(domain.physics_config, 'relative_permeability', 1.0))
                        mu_r_tensor[mask] = mu_val

        mu_r_grid = mu_r_tensor.detach().cpu().numpy().reshape(resolution, resolution)

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        n_levels = getattr(vis_cfg, 'contour_levels', 60)
        
        # --- 1. A_z Plot ---
        az_vmin = vis_cfg.az_vmin if vis_cfg.az_vmin is not None else float(np.min(A_z_grid))
        az_vmax = vis_cfg.az_vmax if vis_cfg.az_vmax is not None else float(np.max(A_z_grid))
        levels_az = np.linspace(az_vmin, az_vmax, n_levels) if az_vmin != az_vmax else n_levels
        c0 = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=levels_az, cmap=vis_cfg.az_cmap)
        fig.colorbar(c0, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
        axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
        axs[0, 0].set_aspect('equal')

        # --- 2. |B| Magnitude Plot ---
        b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
        b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else 1.8
        b_mag_clamped = np.clip(B_mag_grid, b_min, b_max)
        levels_b = np.linspace(b_min, b_max, n_levels)
        c1 = axs[0, 1].contourf(X_grid, Y_grid, b_mag_clamped, levels=levels_b, cmap=vis_cfg.b_mag_cmap, vmin=b_min, vmax=b_max)
        fig.colorbar(c1, ax=axs[0, 1], label=r"$|B|$ (T)")
        axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
        axs[0, 1].set_aspect('equal')

        # --- 3. Relative Permeability \mu_r Plot ---
        v_min_mur = vis_cfg.mur_vmin if vis_cfg.mur_vmin is not None else 0.0
        v_max_mur = vis_cfg.mur_vmax if vis_cfg.mur_vmax is not None else 3000.0
        mu_r_clamped = np.clip(mu_r_grid, v_min_mur, v_max_mur)
        
        if getattr(vis_cfg, 'mur_scale', 'linear') == 'log':
            levels_mur = np.logspace(np.log10(max(v_min_mur, 1e-3)), np.log10(v_max_mur), n_levels)
            norm_mur = LogNorm(vmin=max(v_min_mur, 1.0), vmax=v_max_mur)
            lbl_mur = r"$\mu_r$ (Log Scale)"
        else:
            levels_mur = np.linspace(v_min_mur, v_max_mur, n_levels)
            norm_mur = Normalize(vmin=v_min_mur, vmax=v_max_mur)
            lbl_mur = r"$\mu_r$ (Linear Scale: 0 - 3000)"

        c2 = axs[1, 0].contourf(X_grid, Y_grid, mu_r_clamped, levels=levels_mur, cmap=vis_cfg.mur_cmap, norm=norm_mur)
        fig.colorbar(c2, ax=axs[1, 0], label=lbl_mur)
        axs[1, 0].set_title(r"Relative Permeability Distribution ($\mu_r$)")
        axs[1, 0].set_aspect('equal')

        # --- 4. B_y Component Plot ---
        by_vmin = vis_cfg.by_vmin if vis_cfg.by_vmin is not None else float(np.min(B_y_grid))
        by_vmax = vis_cfg.by_vmax if vis_cfg.by_vmax is not None else float(np.max(B_y_grid))
        levels_by = np.linspace(by_vmin, by_vmax, n_levels) if by_vmin != by_vmax else n_levels
        c3 = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=levels_by, cmap=vis_cfg.by_cmap)
        fig.colorbar(c3, ax=axs[1, 1], label=r"$B_y$ (T)")
        axs[1, 1].set_title(r"Field Component ($B_y$)")
        axs[1, 1].set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axs

    def execute_training_process(self):
        if self.pinn_architecture_instance is None:
            self._build_system()

        def resample_data_fn():
            return self.collocation_sampler_instance.generate_all_domain_points()

        domains_data = resample_data_fn()
        t_cfg = self.training_config

        pre_ep = getattr(t_cfg, 'pretrain_nu_epochs', 300)
        pre_lr = getattr(t_cfg, 'pretrain_nu_lr', 1e-2)
        self.training_manager_instance.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)

        total_adam_epochs = t_cfg.number_of_adam_epochs
        warmup_ratio = float(getattr(t_cfg, 'linear_warmup_epochs_ratio', 0.3))
        epochs_warmup = int(total_adam_epochs * warmup_ratio) if getattr(t_cfg, 'enable_linear_warmup_stage', True) else 0
        epochs_adaptation = (total_adam_epochs - epochs_warmup) if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) else 0

        if getattr(t_cfg, 'enable_linear_warmup_stage', True) and epochs_warmup > 0:
            print(f"\\n--- [STAGE 1A] Adam Linear Warm-up ({epochs_warmup} Epochs) ---")
            domains_data = self.training_manager_instance.train_adam(
                epochs=epochs_warmup, domains_data=domains_data,
                dynamic_resampling=self.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn, resample_interval=self.sampler_config.resample_interval
            )

        if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) and epochs_adaptation > 0:
            fpi_iters = int(getattr(t_cfg, 'fpi_outer_iterations', 10))
            alpha_damp = float(getattr(t_cfg, 'fpi_alpha_damping', 1.0))
            fpi_tol = float(getattr(t_cfg, 'fpi_tolerance', 1e-5))
            epochs_per_fpi = max(1, epochs_adaptation // fpi_iters)

            fit_ep = getattr(t_cfg, 'fit_nu_epochs', 250)
            fit_lr = getattr(t_cfg, 'fit_nu_lr', 3e-3)

            print(f"\\n--- [STAGE 1B] FPI Outer Loop ({fpi_iters} iterations, {epochs_per_fpi} epochs/iter) ---")
            for f_iter in range(fpi_iters):
                print(f"\\n>>> [FPI OUTER LOOP {f_iter + 1}/{fpi_iters}] <<<")
                domains_data = self.training_manager_instance.train_adam(
                    epochs=epochs_per_fpi, domains_data=domains_data,
                    dynamic_resampling=self.sampler_config.dynamic_resampling,
                    resample_fn=resample_data_fn, resample_interval=self.sampler_config.resample_interval
                )
                max_diff = self.training_manager_instance.fit_nu_network_step(
                    domains_data, alpha_damping=alpha_damp, epochs=fit_ep, lr=fit_lr
                )
                print(f"[*] Outer FPI Iter {f_iter + 1}: Max Nu Diff = {max_diff:.4e}")
                if max_diff < fpi_tol:
                    print(f"[✔] FPI Hội tụ sớm tại vòng {f_iter + 1}!")
                    break

        if getattr(t_cfg, 'enable_lbfgs_refinement_stage', True) and t_cfg.number_of_lbfgs_iterations > 0:
            if getattr(self.sampler_config, 'resample_points_before_lbfgs_stage', True):
                domains_data = resample_data_fn()

            lbfgs_iters = int(t_cfg.number_of_lbfgs_iterations)
            print(f"\\n--- [STAGE 2] L-BFGS Refinement ({lbfgs_iters} Iterations) ---")
            self.training_manager_instance.train_lbfgs(max_iterations=lbfgs_iters, domains_data=domains_data)

        if self.visualizer_instance:
            self.visualizer_instance.generate_animations()
'''

    # =========================================================================
    # 5. FIX: utils/training_visualizer.py (Save frame đọc nu từ Net_nu)
    # =========================================================================
    visualizer_code = '''import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import imageio
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

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


class TrainingVisualizer:
    def __init__(self, electro_magnetic_pinn=None, parent_pinn=None):
        self.electro_magnetic_pinn = electro_magnetic_pinn if electro_magnetic_pinn is not None else parent_pinn
        self.parent_pinn = self.electro_magnetic_pinn
        
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        vis_cfg = self.electro_magnetic_pinn.visualization_config
        
        self.output_directory = vis_cfg.output_directory
        self.x_bounds = sim_domain.x_boundaries_tuple
        self.y_bounds = sim_domain.y_boundaries_tuple
        self.resolution = vis_cfg.resolution
        
        self.frame_paths = []
        self.loss_history = []
        self.epoch_history = []
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        x_coords = np.linspace(self.x_bounds[0], self.x_bounds[1], self.resolution)
        y_coords = np.linspace(self.y_bounds[0], self.y_bounds[1], self.resolution)
        self.X_grid, self.Y_grid = np.meshgrid(x_coords, y_coords)
        self.eval_points_tensor = torch.tensor(
            np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel())), 
            dtype=torch.float32
        )

    def save_frame(self, epoch, loss_value, domains_data):
        self.loss_history.append(loss_value)
        self.epoch_history.append(epoch)
        
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        pde_evaluator = self.electro_magnetic_pinn.maxwell_pde_loss_instance
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        vis_cfg = self.electro_magnetic_pinn.visualization_config

        model.eval()
        computation_device = next(model.parameters()).device
        eval_points = self.eval_points_tensor.to(computation_device).detach().clone().requires_grad_(True)
        
        phys_props = sim_domain.evaluate_global_physical_properties(eval_points)
        mat_id = phys_props["material_classification"]
        
        A_z_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_x_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_y_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        nu_phys_all = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        residual_total = torch.zeros((eval_points.shape[0], 1), device=computation_device)

        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, model)
            mask_bool = (mat_id == float(d_id))
            if mask_bool.any():
                Az_sub_star = model.forward_network(net_key, eval_points)
                grad_sub_star = torch.autograd.grad(
                    outputs=Az_sub_star, inputs=eval_points,
                    grad_outputs=torch.ones_like(Az_sub_star), create_graph=True
                )[0]
                Az_sub_phys = Az_sub_star * sim_domain.A0
                Bx_sub_phys = grad_sub_star[:, 1:2] * sim_domain.A0
                By_sub_phys = -grad_sub_star[:, 0:1] * sim_domain.A0

                if getattr(domain.physics_config, 'is_nonlinear', False):
                    # ĐỌC TRỰC TIẾP TỪ MẠNG NET_NU NẾU CÓ
                    if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                        with torch.no_grad():
                            nu_star_pred = model.forward_nu_network(net_key, eval_points)
                            nu_dom = nu_star_pred * sim_domain.nu0
                    else:
                        B_mag_sub = torch.sqrt(Bx_sub_phys**2 + By_sub_phys**2 + 1e-12)
                        nu_dom = domain.evaluate_reluctivity(eval_points, B_magnitude_tensor=B_mag_sub, is_nonlinear=True).to(computation_device)
                else:
                    nu_dom = domain.evaluate_reluctivity(eval_points, is_nonlinear=False).to(computation_device)

                hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
                hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
                jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

                res_pde = pde_evaluator.compute_linear_pde_residual(
                    eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom
                )

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
                nu_phys_all = torch.where(mask_bool, nu_dom.detach(), nu_phys_all)
                residual_total = torch.where(mask_bool, torch.abs(res_pde.detach()), residual_total)
        
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
        mu_r_numpy = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys_all, min=1e-6)).cpu().numpy().reshape(self.resolution, self.resolution)
        residual_numpy = residual_total.cpu().numpy().reshape(self.resolution, self.resolution)
        
        fig, axs = plt.subplots(2, 4, figsize=(23, 10))
        n_levels = getattr(vis_cfg, 'contour_levels', 60)

        # --- SUBPLOT 1: axs[0, 0] ---
        ax_pts = axs[0, 0]
        ax_pts.set_title("Categorized Collocation Points")
        domain_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
        pair_colors = ['#d62728', '#e377c2', '#bcbd22', '#17becf', '#ff1493']

        domains_dict = domains_data.get("domains", domains_data)
        shared_dict = domains_data.get("shared_interfaces", {})

        for idx, (key, d) in enumerate(domains_dict.items()):
            dom_obj = d["domain_object"]
            d_name = getattr(dom_obj, "domain_name", key)
            c_dom = domain_colors[idx % len(domain_colors)]
            
            int_pts = d.get("interior_points")
            if int_pts is not None and int_pts.shape[0] > 0:
                pts_np = int_pts.detach().cpu().numpy()
                ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=2.5, color=c_dom, alpha=0.35, label=f"Int: {d_name}")

        for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
            if_pts = pair_data.get("interface_points")
            if if_pts is not None and if_pts.shape[0] > 0:
                pts_np = if_pts.detach().cpu().numpy()
                c_pair = pair_colors[p_idx % len(pair_colors)]
                name_a = getattr(pair_data["domain_a"], "domain_name", pair_data["key_a"])
                name_b = getattr(pair_data["domain_b"], "domain_name", pair_data["key_b"])
                ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=20, color=c_pair, marker='x', linewidths=1.5, label=f"Interface: {name_a} ↔ {name_b}")

        ax_pts.legend(loc='upper right', fontsize=6.5, framealpha=0.85)
        ax_pts.set_xlim(self.x_bounds)
        ax_pts.set_ylim(self.y_bounds)
        ax_pts.set_aspect('equal')
        
        # --- SUBPLOT 2: axs[0, 1] ---
        res_min = vis_cfg.res_vmin if vis_cfg.res_vmin is not None else float(np.min(residual_numpy))
        res_max = vis_cfg.res_vmax if vis_cfg.res_vmax is not None else float(np.max(residual_numpy))

        if getattr(vis_cfg, 'res_scale', 'log') == 'log':
            res_min_clean = max(res_min, 1e-6)
            res_max_clean = max(res_max, res_min_clean * 10.0)
            res_abs = np.clip(np.abs(residual_numpy), res_min_clean, res_max_clean)
            norm_res = LogNorm(vmin=res_min_clean, vmax=res_max_clean)
            levels_res = np.logspace(np.log10(res_min_clean), np.log10(res_max_clean), n_levels)
        else:
            res_abs = residual_numpy
            norm_res = Normalize(vmin=res_min, vmax=res_max)
            levels_res = np.linspace(res_min, res_max, n_levels)

        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, res_abs, levels=levels_res, cmap=vis_cfg.res_cmap, norm=norm_res)
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')

        # --- SUBPLOT 3: axs[0, 2] ---
        mur_min = vis_cfg.mur_vmin if vis_cfg.mur_vmin is not None else 0.0
        mur_max = vis_cfg.mur_vmax if vis_cfg.mur_vmax is not None else 3000.0
        mu_r_clamped = np.clip(mu_r_numpy, mur_min, mur_max)

        if getattr(vis_cfg, 'mur_scale', 'linear') == 'log':
            mur_min_clean = max(mur_min, 1e-3)
            mur_max_clean = max(mur_max, mur_min_clean * 2.0)
            norm_mur = LogNorm(vmin=mur_min_clean, vmax=mur_max_clean)
            levels_mur = np.logspace(np.log10(mur_min_clean), np.log10(mur_max_clean), n_levels)
            lbl_mur = r"$\mu_r$ (Log Scale)"
        else:
            norm_mur = Normalize(vmin=mur_min, vmax=mur_max)
            levels_mur = np.linspace(mur_min, mur_max, n_levels)
            lbl_mur = r"$\mu_r$ (Linear Scale: 0 - 3000)"

        contour_mur = axs[0, 2].contourf(
            self.X_grid, self.Y_grid, mu_r_clamped, levels=levels_mur,
            cmap=vis_cfg.mur_cmap, norm=norm_mur
        )
        fig.colorbar(contour_mur, ax=axs[0, 2], label=lbl_mur)
        axs[0, 2].set_title(r"Permeability Distribution ($\mu_r$)")
        axs[0, 2].set_aspect('equal')
        
        # --- SUBPLOT 4: axs[0, 3] ---
        axs[0, 3].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 3].set_yscale('log')
        axs[0, 3].set_title("Loss Optimization")
        axs[0, 3].set_xlabel("Step / Epoch")
        axs[0, 3].set_ylabel("Loss")
        axs[0, 3].grid(True, which="both", ls="-", alpha=0.2)
        
        # --- SUBPLOT 5: axs[1, 0] ---
        az_min = vis_cfg.az_vmin if vis_cfg.az_vmin is not None else float(np.min(A_z_numpy))
        az_max = vis_cfg.az_vmax if vis_cfg.az_vmax is not None else float(np.max(A_z_numpy))
        levels_az = np.linspace(az_min, az_max, n_levels) if az_min != az_max else n_levels

        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=levels_az, cmap=vis_cfg.az_cmap)
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        # --- SUBPLOT 6: axs[1, 1] ---
        b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
        b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else 1.8
        b_mag_clamped = np.clip(B_mag_numpy, b_min, b_max)
        levels_b = np.linspace(b_min, b_max, n_levels)

        contour_b = axs[1, 1].contourf(
            self.X_grid, self.Y_grid, b_mag_clamped, levels=levels_b,
            cmap=vis_cfg.b_mag_cmap, vmin=b_min, vmax=b_max
        )
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title(f"Flux Density Magnitude ($|B|$ fixed [{b_min}-{b_max}T])")
        axs[1, 1].set_aspect('equal')

        # --- SUBPLOT 7: axs[1, 2] ---
        contour_b_bg = axs[1, 2].contourf(
            self.X_grid, self.Y_grid, b_mag_clamped, levels=levels_b,
            cmap=vis_cfg.b_mag_cmap, alpha=0.3, vmin=b_min, vmax=b_max
        )
        fig.colorbar(contour_b_bg, ax=axs[1, 2], label="|B| (T)")
        step = max(1, self.resolution // 20)
        B_sub = np.maximum(B_mag_numpy[::step, ::step], 1e-12)
        axs[1, 2].quiver(
            self.X_grid[::step, ::step], self.Y_grid[::step, ::step], 
            B_x_numpy[::step, ::step] / B_sub, B_y_numpy[::step, ::step] / B_sub, 
            scale=30.0, color='black', pivot='mid'
        )
        axs[1, 2].set_title("Flux Density Vectors (B)")
        axs[1, 2].set_aspect('equal')

        # --- SUBPLOT 8: axs[1, 3] ---
        mat_id_numpy = mat_id.cpu().numpy().reshape(self.resolution, self.resolution)
        im_mat = axs[1, 3].imshow(mat_id_numpy, extent=[self.x_bounds[0], self.x_bounds[1], self.y_bounds[0], self.y_bounds[1]], origin='lower', interpolation='nearest', cmap='tab10')
        fig.colorbar(im_mat, ax=axs[1, 3], label="Domain ID")
        axs[1, 3].set_title("Material Classification")
        axs[1, 3].set_aspect('equal')
        
        fig.suptitle(f"XPINN Training Process - Step {epoch}", fontsize=16)
        plt.tight_layout()
        
        frame_name = os.path.join(self.output_directory, f"frame_{len(self.frame_paths) + 1:05d}.png")
        plt.savefig(frame_name, dpi=100)
        plt.close(fig)
        
        self.frame_paths.append(frame_name)
        model.train()

    def generate_gif(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'gif_filename', 'training_process.gif')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        if not self.frame_paths:
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = [imageio.imread(f) for f in self.frame_paths if os.path.exists(f)]
        imageio.mimsave(full_gif_path, images, fps=fps, loop=0)

    def generate_mp4(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        codec = getattr(cfg, 'mp4_codec', 'libx264')
        quality = getattr(cfg, 'mp4_quality', 8)

        valid_frames = [f for f in self.frame_paths if os.path.exists(f)]
        if not valid_frames:
            return

        full_mp4_path = os.path.join(self.output_directory, output_filename)
        try:
            writer = imageio.get_writer(full_mp4_path, fps=fps, codec=codec, pixelformat='yuv420p', quality=quality)
            for f in valid_frames:
                writer.append_data(imageio.imread(f))
            writer.close()
        except Exception as exc:
            pass

    def generate_animations(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        if getattr(cfg, 'generate_gif_animation', True):
            self.generate_gif()
        if getattr(cfg, 'generate_mp4_video', True):
            self.generate_mp4()
'''

    files_to_update = {
        forward_net_file: forward_net_code,
        xpinn_arch_file: xpinn_arch_code,
        training_mgr_file: training_mgr_code,
        pinn_core_file: pinn_core_code,
        visualizer_file: visualizer_code
    }

    for file_path, content in files_to_update.items():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật thành công: {os.path.relpath(file_path, project_root)}")


if __name__ == "__main__":
    apply_permeability_and_project_fixes()