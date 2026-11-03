import os
import sys

def get_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "..", "..")),
        "/content/ElectroMagneticPINN/src/physics_informed_model",
        "/content/ElectroMagneticPINN/physics_informed_model",
        os.getcwd()
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "training_engine")):
            return cand
    return os.path.abspath(os.path.join(script_dir, ".."))

def update_electro_magnetic_pinn(base_dir):
    """Cập nhật electro_magnetic_pinn.py với các tham số cấu hình trọng số động."""
    file_path = os.path.join(base_dir, "electro_magnetic_pinn.py")
    if not os.path.exists(file_path):
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    target_str = "self.training_config = SimpleNamespace("
    new_configs = """self.training_config = SimpleNamespace(
            enable_dynamic_weighting=True,
            dynamic_weight_method="grad_norm", # "grad_norm" hoặc "uncertainty"
            dynamic_weight_update_interval=50,
            dynamic_weight_alpha=0.1,
            max_interface_weight_clamp=20.0,
"""

    if target_str in content and "enable_dynamic_weighting" not in content:
        content = content.replace(target_str, new_configs, 1)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật cấu hình trọng số động trong: {file_path}")

def update_training_manager(base_dir):
    """Bổ sung class AutomaticLossWeighter và hàm update_dynamic_loss_weights vào training_manager.py."""
    file_path = os.path.join(base_dir, "training_engine", "training_manager.py")
    if not os.path.exists(file_path):
        return

    code_content = '''import time
import torch
import torch.nn as nn
import torch.optim as optim

from training_engine.resolve_domain_net_key import resolve_domain_net_key
from training_engine.pretrain_nu_networks import pretrain_nu_networks
from training_engine.fit_nu_network_step import fit_nu_network_step
from training_engine.compute_loss import compute_loss
from training_engine.train_adam import train_adam
from training_engine.train_lbfgs import train_lbfgs


class AutomaticLossWeighter(nn.Module):
    def __init__(self, num_losses=2):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num_losses, requires_grad=True))

    def forward(self, loss_pde, loss_interface):
        p_pde = 0.5 * torch.exp(-self.params[0])
        p_if = 0.5 * torch.exp(-self.params[1])
        return p_pde * loss_pde + p_if * loss_interface + 0.5 * (self.params[0] + self.params[1])


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

        # Khởi tạo loss_weighter nếu chọn phương pháp Uncertainty
        if getattr(t_cfg, 'enable_dynamic_weighting', False) and getattr(t_cfg, 'dynamic_weight_method', '') == "uncertainty":
            self.loss_weighter = AutomaticLossWeighter(num_losses=2).to(self.electro_magnetic_pinn.computation_device)
            self.optimizer_adam = optim.Adam(
                list(model.networks.parameters()) + list(self.loss_weighter.parameters()),
                lr=self.base_learning_rate_adam
            )
        else:
            self.loss_weighter = None
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
        self.last_unweighted_pde_tensor = None
        self.last_unweighted_if_tensor = None

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

    def update_dynamic_loss_weights(self):
        """Thuật toán Cân bằng Gradient Norm (Grad-Norm Balancing)."""
        if self.last_unweighted_pde_tensor is None or self.last_unweighted_if_tensor is None:
            return

        model = self.model
        t_cfg = self.electro_magnetic_pinn.training_config
        alpha = getattr(t_cfg, 'dynamic_weight_alpha', 0.1)
        max_clamp = getattr(t_cfg, 'max_interface_weight_clamp', 20.0)

        # Lấy tham số của lớp cuối cùng trong mạng chính Net_Az
        last_param = None
        for p in model.networks.parameters():
            if p.requires_grad:
                last_param = p

        if last_param is me None or last_param.grad is None:
            return

        grad_pde = torch.autograd.grad(self.last_unweighted_pde_tensor, last_param, retain_graph=True)[0]
        grad_if = torch.autograd.grad(self.last_unweighted_if_tensor, last_param, retain_graph=True)[0]

        norm_pde = torch.mean(torch.abs(grad_pde))
        norm_if = torch.mean(torch.abs(grad_if))

        w_target = norm_pde / (norm_if + 1e-8)

        # Cập nhật mượt trọng số interface cho các domain
        for domain in self.electro_magnetic_pinn.simulation_domain_engine_instance.domains_list:
            if getattr(domain, 'domain_name', '') != 'background':
                w_curr = float(getattr(domain.training_config, 'interface_weight', 1.0))
                w_new = (1.0 - alpha) * w_curr + alpha * w_target.item()
                domain.training_config.interface_weight = min(max(w_new, 0.1), max_clamp)

    def pretrain_nu_networks(self, domains_data, epochs=None, lr=None):
        return pretrain_nu_networks(self, domains_data, epochs=epochs, lr=lr)

    def fit_nu_network_step(self, domains_data, alpha_damping=None, epochs=None, lr=None):
        return fit_nu_network_step(self, domains_data, alpha_damping=alpha_damping, epochs=epochs, lr=lr)

    def compute_loss(self, domains_data):
        return compute_loss(self, domains_data)

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        return train_adam(self, epochs, domains_data, dynamic_resampling=dynamic_resampling, resample_fn=resample_fn, resample_interval=resample_interval)

    def train_lbfgs(self, max_iterations, domains_data):
        return train_lbfgs(self, max_iterations, domains_data)
'''
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code_content)
    print(f"[✔] Đã cập nhật training_manager.py tại: {file_path}")

def update_compute_loss(base_dir):
    """Cập nhật compute_loss.py để lưu trữ tensor unweighted loss phục vụ tính grad_norm."""
    file_path = os.path.join(base_dir, "training_engine", "compute_loss.py")
    if not os.path.exists(file_path):
        return

    code_content = '''import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def compute_loss(training_manager_instance, domains_data):
    model = training_manager_instance.model
    device = next(model.parameters()).device
    dummy_param = next(model.parameters())
    total_loss = (dummy_param * 0.0).sum()

    pde_loss_sum = 0.0
    az_if_sum = 0.0
    bn_if_sum = 0.0
    ht_if_sum = 0.0
    weighted_if_sum = 0.0

    unweighted_pde_tensor = (dummy_param * 0.0).sum()
    unweighted_if_tensor = (dummy_param * 0.0).sum()

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    pinn = training_manager_instance.electro_magnetic_pinn
    bg_domain = pinn.simulation_domain_engine_instance.background_domain
    t_cfg = getattr(pinn, 'training_config', None)

    for key, data in domains_dict.items():
        domain = data["domain_object"]
        net_key = resolve_domain_net_key(domain, model)
        pts = data["interior_points"].to(device)
        
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)
                
            A_z_star = model.forward_network(net_key, pts)
            
            if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                nu_star_pred = model.forward_nu_network(net_key, pts)
                nu_star_net = nu_star_pred if nu_star_pred is not None else None
            else:
                nu_star_net = None

            nu_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
            hx, hy = domain.evaluate_magnetization_vector(pts)
            hx, hy = hx.to(device), hy.to(device)
            jz = domain.evaluate_current_density(pts).to(device)
            
            res_pde = training_manager_instance.pde_evaluator.compute_linear_pde_residual(
                xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_star_net=nu_star_net
            )
            fading_w = training_manager_instance.pde_evaluator.compute_fading_weight(domain, pts)
            
            loss_pde_domain = torch.mean((res_pde ** 2) * fading_w)
            unweighted_pde_tensor = unweighted_pde_tensor + loss_pde_domain

            w_pde = float(getattr(domain.training_config, 'pde_weight', 1.0))
            scaled_pde_loss = w_pde * loss_pde_domain
            total_loss = total_loss + scaled_pde_loss
            pde_loss_sum += scaled_pde_loss.item()

        outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
        if outer_pts.shape[0] > 0 and net_key != 'background':
            if not outer_pts.requires_grad:
                outer_pts = outer_pts.detach().clone().requires_grad_(True)
                
            w_if_loss, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
            )
            unweighted_if_tensor = unweighted_if_tensor + (val_l + bn_l + ht_l)
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

            w_shared_loss, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts
            )
            unweighted_if_tensor = unweighted_if_tensor + (val_l + bn_l + ht_l)
            scaled_shared_loss = w_shared_loss
            total_loss = total_loss + scaled_shared_loss

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += scaled_shared_loss.item()

    # Lưu lại tensor loss phục vụ tính Grad-Norm
    training_manager_instance.last_unweighted_pde_tensor = unweighted_pde_tensor
    training_manager_instance.last_unweighted_if_tensor = unweighted_if_tensor

    # Nếu dùng phương pháp Uncertainty
    if getattr(t_cfg, 'enable_dynamic_weighting', False) and training_manager_instance.loss_weighter is not None:
        total_loss = training_manager_instance.loss_weighter(unweighted_pde_tensor, unweighted_if_tensor)

    training_manager_instance.last_loss_breakdown = {
        "pde_loss": pde_loss_sum,
        "interface_Az": az_if_sum,
        "interface_Bn": bn_if_sum,
        "interface_Ht": ht_if_sum,
        "weighted_interface": weighted_if_sum,
        "total": total_loss.item()
    }

    return total_loss
'''
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code_content)
    print(f"[✔] Đã cập nhật compute_loss.py tại: {file_path}")

def update_train_adam(base_dir):
    """Bổ sung logic gọi update_dynamic_loss_weights theo chu kỳ epoch vào train_adam.py."""
    file_path = os.path.join(base_dir, "training_engine", "train_adam.py")
    if not os.path.exists(file_path):
        return

    code_content = '''import torch
import torch.optim as optim

def train_adam(training_manager_instance, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
    model = training_manager_instance.model
    model.train()
    pinn = training_manager_instance.electro_magnetic_pinn
    t_cfg = getattr(pinn, 'training_config', None)

    eta_min = getattr(t_cfg, 'adam_eta_min', 1e-6)
    max_clip_norm = getattr(t_cfg, 'grad_clip_max_norm', 1.0)
    max_non_finite_streak = getattr(t_cfg, 'max_non_finite_streak', 5)
    log_interval = getattr(t_cfg, 'adam_log_interval', 100)

    dynamic_interval = getattr(t_cfg, 'dynamic_weight_update_interval', 50)
    enable_dynamic = getattr(t_cfg, 'enable_dynamic_weighting', False)
    dynamic_method = getattr(t_cfg, 'dynamic_weight_method', 'grad_norm')

    for param_group in training_manager_instance.optimizer_adam.param_groups:
        param_group['initial_lr'] = training_manager_instance.base_learning_rate_adam
        param_group['lr'] = training_manager_instance.base_learning_rate_adam
        
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(training_manager_instance.optimizer_adam, T_max=max(epochs, 1), eta_min=eta_min)
    stop_reason = f"Hoàn thành toàn bộ {epochs} epochs."
    non_finite_streak = 0
    training_manager_instance.completed_adam_epochs = 0
    
    for epoch in range(epochs):
        if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
            domains_data = resample_fn()

        # Tự động cập nhật trọng số động theo Grad-Norm Balancing theo chu kỳ
        if enable_dynamic and dynamic_method == "grad_norm" and epoch > 0 and epoch % dynamic_interval == 0:
            training_manager_instance.update_dynamic_loss_weights()

        training_manager_instance.optimizer_adam.zero_grad(set_to_none=True)
        loss = training_manager_instance.compute_loss(domains_data)
        
        if not torch.isfinite(loss):
            non_finite_streak += 1
            if non_finite_streak >= max_non_finite_streak:
                stop_reason = f"Dừng sớm do Loss không hữu hạn (NaN/Inf) liên tiếp {max_non_finite_streak} lần."
                break
            continue
        non_finite_streak = 0
            
        loss.backward()
        if max_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.networks.parameters(), max_norm=max_clip_norm)
        training_manager_instance.optimizer_adam.step()
        scheduler_adam.step()
        
        training_manager_instance.completed_adam_epochs = epoch + 1
        current_loss_value = loss.item()
        
        if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0:
            bd = training_manager_instance.last_loss_breakdown
            print(
                f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
            )
            
        if training_manager_instance.visualizer and (epoch + 1) % training_manager_instance.visualizer_update_interval == 0:
            training_manager_instance.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
            
    print(f"[*] KẾT THÚC ADAM. Lý do: {stop_reason}")
    return domains_data
'''
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code_content)
    print(f"[✔] Đã cập nhật train_adam.py tại: {file_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Tiến hành nâng cấp tính năng Self-Adaptive Loss Weighting tại: {base_dir}\n")
    
    update_electro_magnetic_pinn(base_dir)
    update_training_manager(base_dir)
    update_compute_loss(base_dir)
    update_train_adam(base_dir)
    
    print("\n[✔] HOÀN TẤT NÂNG CẤP MÃ NGUỒN! Bạn có thể khởi chạy huấn luyện thử nghiệm.")

if __name__ == "__main__":
    main()