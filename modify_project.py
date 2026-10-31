import os

def update_project_with_pretrain_and_modular_configs():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    mat_domain_file = os.path.join(
        project_root, 'simulation_domain_engine', 'material_domain', 'material_domain.py'
    )
    arch_dir = os.path.join(project_root, 'architectures')
    training_dir = os.path.join(project_root, 'training_engine')
    test_sim_file = os.path.join(project_root, 'tests', 'test_simulation.py')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: material_domain.py (Thêm nu_network_config & update_material_domain)
    # =========================================================================
    files_to_update[mat_domain_file] = '''from typing import Tuple, Union, List, Optional, Any
from types import SimpleNamespace
import torch
import torch.nn as nn

from simulation_domain_engine.material_domain.polygon_signed_distance_field import compute_polygon_signed_distance_field
from simulation_domain_engine.material_domain import assign_material as _assign_material_module
from simulation_domain_engine.material_domain import assign_coercive_direction as _assign_coercive_direction_module
from simulation_domain_engine.material_domain import build_domain as _build_domain_module
from simulation_domain_engine.material_domain import compute_constant_bh_curve as _compute_constant_bh_curve_module
from simulation_domain_engine.material_domain import compute_section_area as _compute_section_area_module
from simulation_domain_engine.material_domain import compute_current_density as _compute_current_density_module
from simulation_domain_engine.material_domain import evaluate_reluctivity as _evaluate_reluctivity_module
from simulation_domain_engine.material_domain import evaluate_magnetization_vector as _evaluate_magnetization_vector_module
from simulation_domain_engine.material_domain import evaluate_current_density as _evaluate_current_density_module

class MaterialDomain:

    def __init__(self, domain_name: str = "air_domain", simulation_domain: Optional[Any] = None) -> None:
        self.simulation_domain = simulation_domain
        self.domain_id: Optional[int] = None
        self.domain_name = domain_name
        
        self.geometry_config = SimpleNamespace(
            outline=None
        )
        
        self.physics_config = SimpleNamespace(
            relative_permeability=1.0,
            coercive=[0.0, 0.0],
            current=0.0,
            bh_curve_data=None,
            is_nonlinear=False
        )
        
        self.training_config = SimpleNamespace(
            pde_weight=1.0,
            pde_nonlinear_weight=1.0,
            interface_weight=1.0,
            interface_potential_weight=1000.0,
            interface_bn_weight=500.0,
            interface_ht_weight=500.0,
            pde_fading_beta=100.0,
            sampling_density_multiplier=1.0,
            number_of_interior_points=2000,
            number_of_surface_points=1200
        )
        
        # Cấu hình Mạng chính (Net_Az)
        self.network_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU(),
            fourier_features=True,
            fourier_sigma=1.2,
            fourier_mapping_size=128
        )

        # Cấu hình Mạng phụ (Net_nu) cho miền phi tuyến
        self.nu_network_config = SimpleNamespace(
            hidden_layers=2,
            hidden_neurons=32,
            activation_function=nn.SiLU(),
            fourier_features=False,
            fourier_sigma=1.0,
            fourier_mapping_size=32
        )
        
        self.outline_tensor: Optional[torch.Tensor] = None
        self.section_area: float = 0.0
        self.current_density: Union[float, Any] = 0.0
        self.coercive_magnitude: float = 0.0

        self.precomputed_edge_starts: Optional[torch.Tensor] = None
        self.precomputed_edge_vecs: Optional[torch.Tensor] = None
        self.precomputed_edge_lens: Optional[torch.Tensor] = None
        self.precomputed_unit_tangents: Optional[torch.Tensor] = None
        self.precomputed_unit_normals: Optional[torch.Tensor] = None

    def assign_material(self, material_name: str = "air") -> "MaterialDomain":
        return _assign_material_module.assign_material(self, material_name=material_name)

    def assign_coercive_direction(self, direction: Tuple[float, float] = (0, 1)) -> "MaterialDomain":
        return _assign_coercive_direction_module.assign_coercive_direction(self, direction=direction)

    def build_domain(self) -> "MaterialDomain":
        return _build_domain_module.build_domain(self)

    def update_material_domain(self) -> "MaterialDomain":
        """
        Đồng bộ toàn bộ thuộc tính cấu hình và dựng lại ranh giới hình học cho miền vật liệu.
        """
        return self.build_domain()

    def _compute_constant_bh_curve(self) -> Any:
        return _compute_constant_bh_curve_module.compute_constant_bh_curve(self)

    def _compute_section_area(self) -> float:
        return _compute_section_area_module.compute_section_area(self)

    def _compute_current_density(self) -> Union[float, Any]:
        return _compute_current_density_module.compute_current_density(self)

    def compute_signed_distance_field(self, points_tensor: torch.Tensor) -> torch.Tensor:
        if self.outline_tensor is None:
            raise RuntimeError(f"Domain '{self.domain_name}' chưa được gọi hàm build_domain().")
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(
        self,
        points_tensor: torch.Tensor,
        B_magnitude_tensor: Optional[torch.Tensor] = None,
        is_nonlinear: Optional[bool] = None
    ) -> torch.Tensor:
        return _evaluate_reluctivity_module.evaluate_reluctivity(
            self, points_tensor, B_magnitude_tensor=B_magnitude_tensor, is_nonlinear=is_nonlinear
        )

    def evaluate_magnetization_vector(self, points_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _evaluate_magnetization_vector_module.evaluate_magnetization_vector(self, points_tensor)

    def evaluate_current_density(self, points_tensor: torch.Tensor) -> torch.Tensor:
        return _evaluate_current_density_module.evaluate_current_density(self, points_tensor)
'''

    # =========================================================================
    # 2. CẬP NHẬT: forward_network.py (Softplus + 1e-6 & Tra cứu qua Net_nu)
    # =========================================================================
    files_to_update[os.path.join(arch_dir, 'forward_network.py')] = '''import torch
import torch.nn as nn
import numpy as np

def forward_network(self, network_key, xy):
    if network_key not in self.networks:
        raise ValueError(f"Lỗi: Không tìm thấy mạng con '{network_key}' trong XPINNArchitecture.")
        
    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == network_key:
            target_domain = domain
            break
        if f"domain_{domain.domain_id}" == network_key:
            target_domain = domain
            break
            
    if target_domain is None:
        raise ValueError(f"Lỗi: Không tìm thấy domain tương ứng với network_key '{network_key}'.")

    config = target_domain.network_config
    
    if config.fourier_features:
        B_matrix = getattr(self, f'B_matrix_{network_key}')
        if B_matrix.device != xy.device:
            B_matrix = B_matrix.to(xy.device)
        projected = (2.0 * np.pi * xy_normalized) @ B_matrix.T
        x_input = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
    else:
        x_input = xy_normalized
        
    raw_output = self.networks[network_key](x_input)
    
    if network_key == 'background':
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
    Được cấu hình thông qua nu_network_config của từng MaterialDomain.
    """
    if network_key not in self.nu_networks:
        return None

    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == network_key or f"domain_{domain.domain_id}" == network_key:
            target_domain = domain
            break

    nu_config = getattr(target_domain, 'nu_network_config', None) if target_domain else None

    if nu_config and getattr(nu_config, 'fourier_features', False):
        B_matrix_nu = getattr(self, f'B_matrix_nu_{network_key}')
        if B_matrix_nu.device != xy.device:
            B_matrix_nu = B_matrix_nu.to(xy.device)
        projected = (2.0 * np.pi * xy_normalized) @ B_matrix_nu.T
        x_input = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
    else:
        x_input = xy_normalized

    raw_nu = self.nu_networks[network_key](x_input)
    # Softplus + 1e-6 đảm bảo nu* luôn dương và hỗ trợ giá trị nhỏ như nu* = 1/1000 = 0.001
    nu_star = torch.nn.functional.softplus(raw_nu) + 1e-6
    return nu_star
'''

    # =========================================================================
    # 3. CẬP NHẬT: xpinn_architecture.py (Khởi tạo Net_nu dựa trên nu_network_config)
    # =========================================================================
    files_to_update[os.path.join(arch_dir, 'xpinn_architecture.py')] = '''import torch
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
                
            # 1. Dựng Mạng chính (Net_Az) từ network_config
            config = domain.network_config
            if config.fourier_features:
                actual_input_dim = config.fourier_mapping_size * 2
                B_mat = torch.randn((config.fourier_mapping_size, self.input_dim)) * config.fourier_sigma
                self.register_buffer(f'B_matrix_{net_key}', B_mat)
            else:
                actual_input_dim = self.input_dim
                
            sub_net = self._build_single_network(actual_input_dim, config)
            self.networks[net_key] = sub_net

            # 2. Dựng Mạng phụ (Net_nu) từ nu_network_config cho miền phi tuyến
            is_nonlinear_domain = getattr(domain.physics_config, 'is_nonlinear', False) or \
                                  (getattr(domain.physics_config, 'bh_curve_data', None) is not None)
            
            if is_nonlinear_domain and net_key != 'background':
                nu_config = getattr(domain, 'nu_network_config', SimpleNamespace(
                    hidden_layers=2, hidden_neurons=32, activation_function=nn.SiLU(), fourier_features=False
                ))
                
                if getattr(nu_config, 'fourier_features', False):
                    actual_nu_input_dim = nu_config.fourier_mapping_size * 2
                    B_mat_nu = torch.randn((nu_config.fourier_mapping_size, self.input_dim)) * nu_config.fourier_sigma
                    self.register_buffer(f'B_matrix_nu_{net_key}', B_mat_nu)
                else:
                    actual_nu_input_dim = self.input_dim

                self.nu_networks[net_key] = self._build_single_network(actual_nu_input_dim, nu_config)
                print(
                    f"[*] XPINNArchitecture: Đã khởi tạo Mạng phụ $Net_{{\\nu}}$ cho miền '{domain.domain_name}' ({net_key}) "
                    f"[{nu_config.hidden_layers} layers x {nu_config.hidden_neurons} neurons]."
                )
            
        self._initialize_weights()
'''

    # =========================================================================
    # 4. CẬP NHẬT: training_manager.py (Pre-train Net_nu & Tra cứu trong compute_loss)
    # =========================================================================
    files_to_update[os.path.join(training_dir, 'training_manager.py')] = '''import time
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
        Huấn luyện ấm trước Mạng phụ Net_nu để khớp hoàn toàn với độ từ kháng ban đầu nu_init*.
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

    def compute_loss(self, domains_data):
        device = next(self.model.parameters()).device
        dummy_param = next(self.model.parameters())
        total_loss = (dummy_param * 0.0).sum()

        pde_loss_sum = 0.0
        mat_loss_sum = 0.0
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
        # Chạy Pre-train Mạng phụ Net_nu trước khi chạy Adam chính
        self.pretrain_nu_networks(domains_data, epochs=300)

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
'''

    # =========================================================================
    # 5. CẬP NHẬT: test_simulation.py (Gọi build_domain() chuẩn hóa)
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

def run_linear_simulation_test():
    output_dir = "figure_linear_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN - PRE-TRAIN NET_NU & TRA CỨU ĐỘ TỪ THẨM MẠNG PHỤ...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. Cấu hình chuẩn hóa trường vật lý
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    # 2. Cấu hình Lấy mẫu động
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # 3. CẤU HÌNH THUẦN TUYẾN TÍNH & HỘI TỤ SÂU
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = False
    model.training_config.enable_lbfgs_refinement_stage = True

    model.training_config.number_of_adam_epochs = 3500
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-9

    model.training_config.number_of_lbfgs_iterations = 1200
    model.training_config.lbfgs_learning_rate = 0.8
    model.training_config.lbfgs_maximum_evaluations = 2000
    model.training_config.lbfgs_tolerance_gradient = 1e-12
    model.training_config.lbfgs_tolerance_change = 1e-12
    model.training_config.lbfgs_history_size = 100

    # 4. Trực quan hóa
    model.visualization_config.active = True
    model.visualization_config.update_interval = 250
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80

    # -------------------------------------------------------------------------
    # 1. Ách sắt (Iron Yoke - Phi tuyến)
    # -------------------------------------------------------------------------
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name="steel_1008",
        outline=[
            [-0.06, -0.03], 
            [0.06, -0.03], 
            [0.06, 0.0], 
            [-0.06, 0.0]
        ]
    )
    iron_yoke.physics_config.is_nonlinear = False
    iron_yoke.training_config.pde_weight = 1.0
    iron_yoke.training_config.interface_weight = 1.5
    iron_yoke.training_config.interface_potential_weight = 1000.0
    iron_yoke.training_config.interface_bn_weight = 500.0
    iron_yoke.training_config.interface_ht_weight = 500.0
    iron_yoke.training_config.pde_fading_beta = 100.0
    iron_yoke.training_config.number_of_interior_points = 4000
    iron_yoke.training_config.number_of_surface_points = 2000

    # Cấu hình Mạng chính Net_Az
    iron_yoke.network_config.hidden_layers = 4
    iron_yoke.network_config.hidden_neurons = 64
    iron_yoke.network_config.activation_function = nn.SiLU()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.2
    iron_yoke.network_config.fourier_mapping_size = 128

    # Cấu hình Mạng phụ Net_nu
    iron_yoke.nu_network_config.hidden_layers = 2
    iron_yoke.nu_network_config.hidden_neurons = 32
    iron_yoke.nu_network_config.activation_function = nn.SiLU()
    iron_yoke.nu_network_config.fourier_features = False

    iron_yoke.build_domain()

    # -------------------------------------------------------------------------
    # 2. Nam châm trái (N30UH - Tuyến tính)
    # -------------------------------------------------------------------------
    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left",
        material_name="N30UH",
        outline=[
            [-0.05, 0.0], 
            [-0.005, 0.0], 
            [-0.005, 0.018], 
            [-0.05, 0.018]
        ],
        coercive_direction=(0, 1)
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.pde_weight = 1.0
    magnet_left.training_config.interface_weight = 1.5
    magnet_left.training_config.interface_potential_weight = 1000.0
    magnet_left.training_config.interface_bn_weight = 500.0
    magnet_left.training_config.interface_ht_weight = 500.0
    magnet_left.training_config.pde_fading_beta = 100.0
    magnet_left.training_config.number_of_interior_points = 2500
    magnet_left.training_config.number_of_surface_points = 1500

    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.SiLU()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.2
    magnet_left.network_config.fourier_mapping_size = 128
    magnet_left.build_domain()

    # -------------------------------------------------------------------------
    # 3. Nam châm phải (N30UH - Tuyến tính)
    # -------------------------------------------------------------------------
    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right",
        material_name="N30UH",
        outline=[
            [0.005, 0.0], 
            [0.05, 0.0], 
            [0.05, 0.018], 
            [0.005, 0.018]
        ],
        coercive_direction=(0, -1)
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.pde_weight = 1.0
    magnet_right.training_config.interface_weight = 1.5
    magnet_right.training_config.interface_potential_weight = 1000.0
    magnet_right.training_config.interface_bn_weight = 500.0
    magnet_right.training_config.interface_ht_weight = 500.0
    magnet_right.training_config.pde_fading_beta = 100.0
    magnet_right.training_config.number_of_interior_points = 2500
    magnet_right.training_config.number_of_surface_points = 1500

    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.SiLU()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.2
    magnet_right.network_config.fourier_mapping_size = 128
    magnet_right.build_domain()

    # -------------------------------------------------------------------------
    # 4. Background Air Domain
    # -------------------------------------------------------------------------
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        interface_weight=1.5,
        pde_fading_beta=100.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=10000,
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
    sim_domain.plot_problem_definition(resolution=100)
    
    print("[*] Chạy tiến trình huấn luyện tuyến tính...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ...")
    model.plot_fields(
        resolution=180, 
        save_path=f"{output_dir}/linear_final_fields.png"
    )

    print(f"[✔] HOÀN THÀNH KIỂM THỬ! Kết quả lưu tại {output_dir}/")

def test_simulation4():
    run_linear_simulation_test()

if __name__ == "__main__":
    run_linear_simulation_test()
'''

    # Ghi đè file
    for path, code in files_to_update.items():
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code.strip() + '\n')
        print(f"[✔] Đã áp dụng cập nhật cho tệp: {os.path.relpath(path, project_root)}")

if __name__ == "__main__":
    update_project_with_pretrain_and_modular_configs()
