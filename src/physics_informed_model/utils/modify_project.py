import os

def apply_all_fixes():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    maxwell_pde_file = os.path.join(project_root, 'physics_domain', 'physical_equations', 'maxwell_pde_loss.py')
    training_manager_file = os.path.join(project_root, 'training_engine', 'training_manager.py')
    test_sim_file = os.path.join(project_root, 'tests', 'test_simulation.py')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: training_manager.py
    # Sửa lỗi ngắt Autograd Graph và tối ưu hóa theo Batch cho Net_nu
    # =========================================================================
    files_to_update[training_manager_file] = '''import time
import torch
import torch.optim as optim

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

        print(f"\\n[*] Pre-training Mạng phụ $Net={{\\\\nu}}$ về hằng số ban đầu ({epochs} epochs, lr={lr})...")
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
                if key in self.model.nu_networks:
                    domain = data["domain_object"]
                    pts = data["interior_points"].to(device)
                    
                    if pts.shape[0] > 0:
                        nu_pred_star = self.model.forward_nu_network(key, pts)
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

        # 1. Tính toán nhãn mục tiêu FPI cho các miền phi tuyến
        for key, data in domains_dict.items():
            if key in self.model.nu_networks:
                domain = data["domain_object"]
                if not getattr(domain.physics_config, 'is_nonlinear', False):
                    continue

                pts = data["interior_points"].to(device)
                if pts.shape[0] > 0:
                    pts_grad = pts.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        Az_star = self.model.forward_network(key, pts_grad)
                        grad_Az = torch.autograd.grad(
                            outputs=Az_star, inputs=pts_grad,
                            grad_outputs=torch.ones_like(Az_star), create_graph=False
                        )[0]
                        Bx_star = grad_Az[:, 1:2] * sim_domain.L0
                        By_star = -grad_Az[:, 0:1] * sim_domain.L0
                        B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * sim_domain.B0
                        
                    with torch.no_grad():
                        nu_BH_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys.detach(), is_nonlinear=True)
                        nu_BH_star = nu_BH_phys / sim_domain.nu0
                        
                        nu_old_star = self.model.forward_nu_network(key, pts)
                        nu_target_star = ((1.0 - alpha_damping) * nu_old_star + alpha_damping * nu_BH_star).detach()
                        
                        diff_val = torch.max(torch.abs(nu_target_star - nu_old_star)).item()
                        max_nu_diff = max(max_nu_diff, diff_val)
                        targets_dict[key] = (pts, nu_target_star)

        if not targets_dict:
            return 0.0

        # 2. Khớp đồng thời Mạng Net_nu trên tất cả các miền phi tuyến
        nu_params = []
        for key in targets_dict.keys():
            nu_params.extend(self.model.nu_networks[key].parameters())
        opt_nu = optim.Adam(nu_params, lr=lr)

        for _ in range(epochs):
            opt_nu.zero_grad()
            total_fit_loss = torch.tensor(0.0, device=device)
            for key, (pts, nu_target_star) in targets_dict.items():
                nu_pred_star = self.model.forward_nu_network(key, pts)
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

        # 1. PDE Loss
        for key, data in domains_dict.items():
            domain = data["domain_object"]
            pts = data["interior_points"].to(device)
            
            if pts.shape[0] > 0:
                if not pts.requires_grad:
                    pts = pts.detach().clone().requires_grad_(True)
                    
                A_z_star = self.model.forward_network(key, pts)
                
                # Giữ nguyên Autograd Graph để tính grad(nu*) theo tọa độ không gian xy
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
                    key_a=key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
                )
                scaled_if_loss = w_if_loss
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
    # 2. CẬP NHẬT: test_simulation.py
    # Chuẩn hóa fpi_alpha_damping = 0.25
    # =========================================================================
    files_to_update[test_sim_file] = '''import os
import sys
import shutil
from types import SimpleNamespace

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from electro_magnetic_pinn import ElectroMagneticPINN

def run_nonlinear_simulation_test():
    output_dir = "figure_nonlinear_optimized_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] KHỞI TẠO XPINN PHI TUYẾN - CẤU HÌNH SIÊU THAM SỐ LINH HOẠT...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. CẤU HÌNH KHUNG TOẠ ĐỘ VÀ QUY MÔ VẬT LÝ
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    # 2. CẤU HÌNH LẤY MẪU
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 50
    model.sampler_config.resample_points_before_lbfgs_stage = False

    # 3. TOÀN BỘ CẤU HÌNH THAM SỐ HUẤN LUYỆN
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = True

    # Tổng Epochs & Tốc độ học Adam
    model.training_config.number_of_adam_epochs = 5000
    model.training_config.linear_warmup_epochs_ratio = 0.30       # ~1500 Epochs Warm-up
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-8

    # Cấu hình FPI Outer Loop
    model.training_config.fpi_outer_iterations = 10               # 10 vòng lặp FPI ngoại
    model.training_config.fpi_alpha_damping = 0.25               # Damping factor 0.25 (Đã chuẩn hóa)
    model.training_config.fpi_tolerance = 1e-5                    # Dung sai FPI

    # Cấu hình Khớp Mạng phụ Net_nu
    model.training_config.pretrain_nu_epochs = 300               # Epochs Pre-train Net_nu ban đầu
    model.training_config.pretrain_nu_lr = 1e-2                  # LR Pre-train Net_nu
    model.training_config.fit_nu_epochs = 250                    # Epochs khớp Net_nu mỗi bước FPI
    model.training_config.fit_nu_lr = 3e-3                       # LR khớp Net_nu mỗi bước FPI

    # Kiểm soát Gradient & Nhật ký
    model.training_config.grad_clip_max_norm = 1.0              # Clip gradient max norm
    model.training_config.adam_eta_min = 1e-6                    # LR tối thiểu của Cosine Annealing
    model.training_config.max_non_finite_streak = 5              # Giới hạn NaN/Inf streak
    model.training_config.adam_log_interval = 100               # Tần suất in log Adam (epochs)
    model.training_config.lbfgs_log_interval = 25               # Tần suất in log L-BFGS (closures)

    # Cấu hình L-BFGS Refinement
    model.training_config.number_of_lbfgs_iterations = 1500
    model.training_config.lbfgs_learning_rate = 0.5
    model.training_config.lbfgs_maximum_evaluations = 2500
    model.training_config.lbfgs_tolerance_gradient = 1e-12
    model.training_config.lbfgs_tolerance_change = 1e-12
    model.training_config.lbfgs_history_size = 50

    # 4. TRỰC QUAN HÓA
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 120
    model.visualization_config.generate_gif_animation = False
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "nonlinear_optimized_training.mp4"

    # 5. CẤU HÌNH CÁC MIỀN VẬT LIỆU
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke", material_name="steel_1008",
        outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    )
    iron_yoke.physics_config.is_nonlinear = True
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_potential_weight = 100.0
    iron_yoke.training_config.interface_bn_weight = 50.0
    iron_yoke.training_config.interface_ht_weight = 50.0
    iron_yoke.training_config.pde_fading_beta = 60.0
    iron_yoke.training_config.number_of_interior_points = 4500
    iron_yoke.training_config.number_of_surface_points = 1500

    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128

    iron_yoke.nu_network_config.hidden_layers = 3
    iron_yoke.nu_network_config.hidden_neurons = 48
    iron_yoke.nu_network_config.activation_function = nn.SiLU()
    iron_yoke.nu_network_config.fourier_features = False

    iron_yoke.build_domain()

    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left", material_name="N30UH",
        outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_potential_weight = 100.0
    magnet_left.training_config.interface_bn_weight = 50.0
    magnet_left.training_config.interface_ht_weight = 50.0
    magnet_left.training_config.number_of_interior_points = 2000
    magnet_left.training_config.number_of_surface_points = 1000

    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128
    magnet_left.build_domain()

    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right", material_name="N30UH",
        outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_potential_weight = 100.0
    magnet_right.training_config.interface_bn_weight = 50.0
    magnet_right.training_config.interface_ht_weight = 50.0
    magnet_right.training_config.number_of_interior_points = 2000
    magnet_right.training_config.number_of_surface_points = 1000

    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128
    magnet_right.build_domain()

    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0, 
        pde_fading_beta=60.0,
        sampling_density_multiplier=1.0, 
        number_of_interior_points=8000,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=64,
        activation_function=nn.SiLU(),
        fourier_features=True, 
        fourier_sigma=1.2, 
        fourier_mapping_size=128
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    model.update_electromagnetic_pinn()
    
    print("[*] Chạy tiến trình huấn luyện PHI TUYẾN...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ phi tuyến...")
    model.plot_fields(resolution=200, save_path=f"{output_dir}/nonlinear_final_fields.png")
    print(f"[✔] HOÀN THÀNH KIỂM THỬ PHI TUYẾN! Kết quả lưu tại: {output_dir}/")

if __name__ == "__main__":
    run_nonlinear_simulation_test()
'''

    # Thực thi cập nhật
    for file_path, content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật thành công: {os.path.relpath(file_path, project_root)}")

if __name__ == "__main__":
    apply_all_fixes()