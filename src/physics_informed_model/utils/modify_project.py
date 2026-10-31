import os

def update_entire_xpinn_fpi_project():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    mat_domain_file = os.path.join(project_root, 'simulation_domain_engine', 'material_domain', 'material_domain.py')
    eval_reluc_file = os.path.join(project_root, 'simulation_domain_engine', 'material_domain', 'evaluate_reluctivity.py')
    arch_dir = os.path.join(project_root, 'architectures')
    training_dir = os.path.join(project_root, 'training_engine')
    physics_eq_dir = os.path.join(project_root, 'physics_domain', 'physical_equations')
    main_pinn_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    test_sim_file = os.path.join(project_root, 'tests', 'test_simulation.py')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: material_domain.py (Thêm nu_network_config & fpi_config)
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

        # Cấu hình lặp Fixed-Point Iteration (FPI)
        self.fpi_config = SimpleNamespace(
            relaxation_alpha=0.5,
            max_fpi_iterations=10,
            fpi_tolerance=1e-4,
            use_net_nu=True
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
    # 2. CẬP NHẬT: evaluate_reluctivity.py (Xử lý B -> 0 tiệm cận chuẩn)
    # =========================================================================
    files_to_update[eval_reluc_file] = '''import torch
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

def evaluate_reluctivity(self, points_tensor, B_magnitude_tensor=None, is_nonlinear=None):
    if is_nonlinear is None:
        is_nonlinear = getattr(self.physics_config, 'is_nonlinear', False)

    if not is_nonlinear:
        mu_r = float(getattr(self.physics_config, 'relative_permeability', 1.0))
        const_nu = VACUUM_RELUCTIVITY / max(mu_r, 1e-6)
        return torch.full(
            (points_tensor.shape[0], 1),
            const_nu,
            dtype=points_tensor.dtype,
            device=points_tensor.device
        )

    bh_data = getattr(self.physics_config, 'bh_curve_data', None)
    if isinstance(bh_data, (list, tuple)) and len(bh_data) == 2:
        b_pts = torch.tensor(bh_data[0], dtype=points_tensor.dtype, device=points_tensor.device)
        h_pts = torch.tensor(bh_data[1], dtype=points_tensor.dtype, device=points_tensor.device)

        if B_magnitude_tensor is None:
            B_magnitude_tensor = torch.zeros(
                (points_tensor.shape[0], 1),
                dtype=points_tensor.dtype,
                device=points_tensor.device
            )

        b_flat = B_magnitude_tensor.view(-1)
        indices = torch.bucketize(b_flat, b_pts) - 1
        indices = torch.clamp(indices, 0, len(b_pts) - 2)

        b0, b1 = b_pts[indices], b_pts[indices + 1]
        h0, h1 = h_pts[indices], h_pts[indices + 1]

        t = (b_flat - b0) / (b1 - b0 + 1e-12)
        h_interp = h0 + t * (h1 - h0)

        # Xử lý giới hạn B -> 0 dùng độ dốc ban đầu (H1/B1)
        initial_slope = h_pts[1] / max(float(b_pts[1].item()), 1e-8)
        nu_tensor = torch.where(b_flat > 1e-6, h_interp / b_flat, initial_slope)
        return nu_tensor.view(-1, 1)

    raise ValueError(f"Domain '{self.domain_name}' đang ở chế độ phi tuyến nhưng thiếu dữ liệu bh_curve_data [B, H] hợp lệ.")
'''

    # =========================================================================
    # 3. CẬP NHẬT: forward_network.py (Softplus + 1e-6 cho Net_nu)
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
    nu_star = torch.nn.functional.softplus(raw_nu) + 1e-6
    return nu_star
'''

    # =========================================================================
    # 4. CẬP NHẬT: xpinn_architecture.py (Khởi tạo Mạng kép Az + Nu)
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

            # 2. Dựng Mạng phụ (Net_nu) cho miền phi tuyến
            is_nonlinear_domain = getattr(domain.physics_config, 'is_nonlinear', False) or \\
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
                    f"[*] XPINNArchitecture: Đã khởi tạo Mạng phụ $Net_{{\\\\nu}}$ cho miền '{domain.domain_name}' ({net_key}) "
                    f"[{nu_config.hidden_layers} layers x {nu_config.hidden_neurons} neurons]."
                )
            
        self._initialize_weights()
'''

    # =========================================================================
    # 5. CẬP NHẬT: maxwell_pde_loss.py (PDE Residual dùng nu* từ Net_nu / FPI)
    # =========================================================================
    files_to_update[os.path.join(physics_eq_dir, 'maxwell_pde_loss.py')] = '''import torch

class MaxwellPDELoss:
    def __init__(self, electro_magnetic_pinn=None, L0=0.1, H0=800000.0, nu0=795774.715459):
        self.electro_magnetic_pinn = electro_magnetic_pinn
        self._L0 = L0
        self._H0 = H0
        self._nu0 = nu0

    @property
    def L0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.L0
        return self._L0

    @property
    def H0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.H0
        return self._H0

    @property
    def nu0(self):
        if self.electro_magnetic_pinn is not None:
            return self.electro_magnetic_pinn.simulation_domain_engine_instance.nu0
        return self._nu0

    def compute_linear_pde_residual(self, xy, A_z_star, nu_phys, J_z, H_cx, H_cy, nu_star_net=None):
        """
        Tính dư số PDE Maxwell-Poisson ở chế độ FPI / Tuyến tính hóa:
        curl(H) - Jz = 0  với H = nu * B - Hc
        """
        if nu_star_net is not None:
            nu_star = nu_star_net
        else:
            nu_star = nu_phys / self.nu0

        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)

        # 1. Gradient của Az* -> B*
        grad_A_star = torch.autograd.grad(
            outputs=A_z_star,
            inputs=xy,
            grad_outputs=torch.ones_like(A_z_star),
            create_graph=True
        )[0]

        dAz_dx_star = grad_A_star[:, 0:1] * self.L0
        dAz_dy_star = grad_A_star[:, 1:2] * self.L0

        # 2. H* = nu* * B* - Hc*
        Hx_star = nu_star * dAz_dy_star - H_cx_star
        Hy_star = -nu_star * dAz_dx_star - H_cy_star

        # 3. curl(H*) = dHy*/dx* - dHx*/dy*
        grad_Hx = torch.autograd.grad(outputs=Hx_star, inputs=xy, grad_outputs=torch.ones_like(Hx_star), create_graph=True)[0]
        grad_Hy = torch.autograd.grad(outputs=Hy_star, inputs=xy, grad_outputs=torch.ones_like(Hy_star), create_graph=True)[0]

        dHy_dx_star = grad_Hy[:, 0:1] * self.L0
        dHx_dy_star = grad_Hx[:, 1:2] * self.L0

        res_pde = dHy_dx_star - dHx_dy_star - J_z_star
        nu_scale = torch.clamp(nu_star.detach(), min=1e-6)
        return res_pde / nu_scale

    def compute_fading_weight(self, domain, xy):
        beta = float(getattr(domain.training_config, 'pde_fading_beta', 80.0))
        if beta <= 0.0:
            return torch.ones((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)

        if self.electro_magnetic_pinn is not None:
            sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
            mat_verts = [
                d.outline_tensor.to(device=xy.device, dtype=xy.dtype)
                for d in sim_domain.domains_list
                if getattr(d, 'domain_name', '') != 'background' and d.outline_tensor is not None
            ]
            if mat_verts:
                all_verts = torch.cat(mat_verts, dim=0)
                dists = torch.cdist(xy.detach(), all_verts, p=2.0)
                min_corner_dist, _ = torch.min(dists, dim=1, keepdim=True)
                corner_dist_star = min_corner_dist / self.L0
                return 1.0 - torch.exp(-beta * corner_dist_star)

        sdf = domain.compute_signed_distance_field(xy).view(-1, 1)
        abs_sdf_star = torch.abs(sdf.detach()) / self.L0
        return 1.0 - torch.exp(-beta * abs_sdf_star)
'''

    # =========================================================================
    # 6. CẬP NHẬT: training_manager.py (Trình điều khiển FPI & Pre-train Net_nu)
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
        Huấn luyện ấm Mạng phụ Net_nu để khớp với độ từ kháng ban đầu nu_init*.
        """
        if not hasattr(self.model, 'nu_networks') or len(self.model.nu_networks) == 0:
            return

        print(f"\\n[*] Bắt đầu Pre-training Mạng phụ $Net_{{\\\\nu}}$ ({epochs} epochs)...")
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
                
        print("[✔] Hoàn thành Pre-train Mạng phụ $Net_{\\\\nu}$!")

    def train_fpi(self, max_fpi_iterations, inner_adam_epochs, domains_data, relaxation_alpha=0.5):
        """
        Vòng lặp ngoài Fixed-Point Iteration (FPI/Picard) kết hợp làm mềm relaxation.
        """
        print(f"\\n[*] BẮT ĐẦU VÒNG LẶP FIXED-POINT ITERATION (FPI) [Tối đa {max_fpi_iterations} vòng ngoài]...")
        device = next(self.model.parameters()).device
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance

        for fpi_iter in range(max_fpi_iterations):
            print(f"\\n=================== [FPI OUTER LOOP {fpi_iter + 1}/{max_fpi_iterations}] ===================")
            
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
'''

    # Ghi đè toàn bộ tệp trong dự án
    for path, code in files_to_update.items():
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code.strip() + '\n')
        print(f"[✔] modify_project: Đã cập nhật tệp -> {os.path.relpath(path, project_root)}")

if __name__ == "__main__":
    update_entire_xpinn_fpi_project()