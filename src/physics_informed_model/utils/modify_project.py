import os

def apply_full_optimizations_and_visuals():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    sim_engine_dir = os.path.join(project_root, 'simulation_domain_engine')
    mat_domain_dir = os.path.join(sim_engine_dir, 'material_domain')
    physics_dir = os.path.join(project_root, 'physics_domain', 'physical_equations')
    training_dir = os.path.join(project_root, 'training_engine')
    utils_dir = os.path.join(project_root, 'utils')
    tests_dir = os.path.join(project_root, 'tests')
    pinn_main_file = os.path.join(project_root, 'electro_magnetic_pinn.py')

    files_to_update = {}

    # =========================================================================
    # 1. CẬP NHẬT: simulation_domain_engine/material_domain/material_domain.py
    # =========================================================================
    files_to_update[os.path.join(mat_domain_dir, 'material_domain.py')] = '''from typing import Tuple, Union, List, Optional, Any
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
            pde_fading_beta=150.0,
            sampling_density_multiplier=1.0,
            number_of_interior_points=1000,
            number_of_surface_points=800
        )
        
        self.network_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU(),
            fourier_features=True,
            fourier_sigma=1.2,
            fourier_mapping_size=128
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
    # 2. CẬP NHẬT: simulation_domain_engine/material_domain/build_domain.py
    # =========================================================================
    files_to_update[os.path.join(mat_domain_dir, 'build_domain.py')] = '''import torch

def build_domain(self):
    if self.geometry_config.outline is not None:
        if len(self.geometry_config.outline) < 3:
            raise ValueError(f"Lỗi: Outline của domain '{self.domain_name}' phải có ít nhất 3 đỉnh.")
        
        self.outline_tensor = torch.tensor(self.geometry_config.outline, dtype=torch.float32)
        self._compute_section_area()
        if self.section_area <= 1e-12:
            raise ValueError(f"Lỗi: Diện tích của domain '{self.domain_name}' bằng 0 hoặc suy biến.")
        self._compute_current_density()

        starts = self.outline_tensor
        ends = torch.roll(self.outline_tensor, shifts=-1, dims=0)
        edge_vecs = ends - starts
        edge_lens = torch.sqrt(torch.sum(edge_vecs ** 2, dim=1, keepdim=True) + 1e-12)

        signed_area = 0.5 * torch.sum(starts[:, 0] * ends[:, 1] - ends[:, 0] * starts[:, 1])
        orient = 1.0 if signed_area.item() >= 0.0 else -1.0

        unit_tangents = orient * (edge_vecs / edge_lens)
        unit_normals = torch.cat([unit_tangents[:, 1:2], -unit_tangents[:, 0:1]], dim=1)

        self.precomputed_edge_starts = starts
        self.precomputed_edge_vecs = edge_vecs
        self.precomputed_edge_lens = edge_lens
        self.precomputed_unit_tangents = unit_tangents
        self.precomputed_unit_normals = unit_normals

        coercive_cfg = self.physics_config.coercive
        if isinstance(coercive_cfg, (list, tuple)) and len(coercive_cfg) == 2:
            self.coercive_magnitude = float((coercive_cfg[0]**2 + coercive_cfg[1]**2) ** 0.5)
    else:
        raise ValueError(f"Lỗi: Domain '{self.domain_name}' chưa được cấp outline trong geometry_config.")
        
    if self.physics_config.bh_curve_data is None:
        self._compute_constant_bh_curve()
        
    return self
'''

    # =========================================================================
    # 3. CẬP NHẬT: simulation_domain_engine/material_domain/assign_material.py
    # =========================================================================
    files_to_update[os.path.join(mat_domain_dir, 'assign_material.py')] = '''def assign_material(self, material_name="air"):
    if self.simulation_domain is None or not hasattr(self.simulation_domain, 'material_data'):
        raise RuntimeError(f"Lỗi: Domain '{self.domain_name}' chưa được gán vào SimulationDomain nên không thể tra cứu vật liệu.")

    mat = self.simulation_domain.material_data.get_material(material_name)
    self.physics_config.relative_permeability = mat.relative_permeability
    self.physics_config.coercive = mat.coercive
    self.physics_config.bh_curve_data = mat.bh_curve_data
    
    if mat.bh_curve_data is not None:
        self.physics_config.is_nonlinear = True
    else:
        self.physics_config.is_nonlinear = False
    
    print(f"[*] Domain '{self.domain_name}': Đã gán vật liệu '{mat.name}' (Chế độ phi tuyến: {self.physics_config.is_nonlinear}).")
    return self
'''

    # =========================================================================
    # 4. CẬP NHẬT: simulation_domain_engine/material_domain/evaluate_reluctivity.py
    # =========================================================================
    files_to_update[os.path.join(mat_domain_dir, 'evaluate_reluctivity.py')] = '''import torch
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

        nu_tensor = torch.where(b_flat > 1e-6, h_interp / b_flat, h_pts[1] / b_pts[1])
        return nu_tensor.view(-1, 1)

    raise ValueError(f"Domain '{self.domain_name}' đang ở chế độ phi tuyến nhưng thiếu dữ liệu bh_curve_data [B, H] hợp lệ.")
'''

    # =========================================================================
    # 5. CẬP NHẬT: physics_domain/physical_equations/maxwell_pde_loss.py
    # =========================================================================
    files_to_update[os.path.join(physics_dir, 'maxwell_pde_loss.py')] = '''import torch

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

    def compute_residual_components(self, xy, A_z_star, nu_phys, J_z, H_cx, H_cy, nu_linear_phys):
        nu_star = nu_phys / self.nu0
        nu_lin_star = nu_linear_phys / self.nu0
        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)
        
        grad_A_star = torch.autograd.grad(
            outputs=A_z_star,
            inputs=xy,
            grad_outputs=torch.ones_like(A_z_star),
            create_graph=True
        )[0]
        
        dAz_star_dx_star = grad_A_star[:, 0:1] * self.L0
        dAz_star_dy_star = grad_A_star[:, 1:2] * self.L0
        
        Hx_lin_star = nu_lin_star * dAz_star_dy_star - H_cx_star
        Hy_lin_star = -nu_lin_star * dAz_star_dx_star - H_cy_star
        
        grad_Hx_lin = torch.autograd.grad(outputs=Hx_lin_star, inputs=xy, grad_outputs=torch.ones_like(Hx_lin_star), create_graph=True)[0]
        grad_Hy_lin = torch.autograd.grad(outputs=Hy_lin_star, inputs=xy, grad_outputs=torch.ones_like(Hy_lin_star), create_graph=True)[0]
        
        res_linear_star = grad_Hy_lin[:, 0:1] * self.L0 - grad_Hx_lin[:, 1:2] * self.L0 - J_z_star

        delta_nu_star = nu_star - nu_lin_star
        Hx_nonlin_star = delta_nu_star * dAz_star_dy_star
        Hy_nonlin_star = -delta_nu_star * dAz_star_dx_star
        
        grad_Hx_nonlin = torch.autograd.grad(outputs=Hx_nonlin_star, inputs=xy, grad_outputs=torch.ones_like(Hx_nonlin_star), create_graph=True)[0]
        grad_Hy_nonlin = torch.autograd.grad(outputs=Hy_nonlin_star, inputs=xy, grad_outputs=torch.ones_like(Hy_nonlin_star), create_graph=True)[0]
        
        res_nonlin_star = grad_Hy_nonlin[:, 0:1] * self.L0 - grad_Hx_nonlin[:, 1:2] * self.L0

        nu_scale = torch.clamp(nu_star.detach(), min=1e-6)
        return res_linear_star / nu_scale, res_nonlin_star / nu_scale

    def compute_fading_weight(self, domain, xy):
        beta = float(getattr(domain.training_config, 'pde_fading_beta', 150.0))
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
    # 6. CẬP NHẬT: physics_domain/physical_equations/interface_loss.py
    # =========================================================================
    files_to_update[os.path.join(physics_dir, 'interface_loss.py')] = '''import torch

class InterfaceLoss:
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

    def _compute_normals_and_tangents_from_outline(self, domain, surf_pts):
        start_pts = domain.precomputed_edge_starts.to(device=surf_pts.device, dtype=surf_pts.dtype)
        edge_vecs = domain.precomputed_edge_vecs.to(device=surf_pts.device, dtype=surf_pts.dtype)
        edge_lens = domain.precomputed_edge_lens.to(device=surf_pts.device, dtype=surf_pts.dtype)
        unit_tangents = domain.precomputed_unit_tangents.to(device=surf_pts.device, dtype=surf_pts.dtype)
        unit_normals = domain.precomputed_unit_normals.to(device=surf_pts.device, dtype=surf_pts.dtype)

        pt_to_start = surf_pts.unsqueeze(1) - start_pts.unsqueeze(0)
        denom = (edge_lens.squeeze(1) ** 2 + 1e-12).unsqueeze(0)
        proj = torch.clamp(torch.sum(pt_to_start * edge_vecs.unsqueeze(0), dim=2) / denom, 0.0, 1.0)
        closest = start_pts.unsqueeze(0) + proj.unsqueeze(-1) * edge_vecs.unsqueeze(0)
        dist_sq = torch.sum((surf_pts.unsqueeze(1) - closest) ** 2, dim=2)
        edge_idx = torch.argmin(dist_sq, dim=1)

        t_edge = torch.gather(proj, 1, edge_idx.unsqueeze(1)).detach()
        corner_w = (torch.tanh(12.0 * t_edge) * torch.tanh(12.0 * (1.0 - t_edge))).detach()

        if self.electro_magnetic_pinn is not None:
            sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
            mat_verts = [
                d.outline_tensor.to(device=surf_pts.device, dtype=surf_pts.dtype)
                for d in sim_domain.domains_list
                if getattr(d, 'domain_name', '') != 'background' and d.outline_tensor is not None
            ]
            if mat_verts:
                all_verts = torch.cat(mat_verts, dim=0)
                v_dists = torch.cdist(surf_pts.detach(), all_verts, p=2.0)
                min_v_dist, _ = torch.min(v_dists, dim=1, keepdim=True)
                vertex_w = (torch.tanh(25.0 * (min_v_dist / self.L0)) ** 2).detach()
                corner_w = corner_w * vertex_w

        nx = unit_normals[edge_idx, 0:1].detach()
        ny = unit_normals[edge_idx, 1:2].detach()
        tx = unit_tangents[edge_idx, 0:1].detach()
        ty = unit_tangents[edge_idx, 1:2].detach()
        return nx, ny, tx, ty, corner_w

    def _compute_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            zero = torch.tensor(0.0, device=pts.device)
            return zero, zero, zero

        model = self.electro_magnetic_pinn.pinn_architecture_instance
        nx, ny, tx, ty, corner_w = self._compute_normals_and_tangents_from_outline(domain_a, pts)

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            _, _, _, _, corner_w_b = self._compute_normals_and_tangents_from_outline(domain_b, pts)
            corner_w = torch.minimum(corner_w, corner_w_b)

        Az_a = model.forward_network(key_a, pts)
        Az_b = model.forward_network(key_b, pts)

        loss_val = torch.mean((Az_a - Az_b) ** 2)

        grad_a = torch.autograd.grad(
            outputs=Az_a,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_a),
            create_graph=True
        )[0]

        grad_b = torch.autograd.grad(
            outputs=Az_b,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_b),
            create_graph=True
        )[0]

        Bx_a_star = grad_a[:, 1:2] * self.L0
        By_a_star = -grad_a[:, 0:1] * self.L0
        Bx_b_star = grad_b[:, 1:2] * self.L0
        By_b_star = -grad_b[:, 0:1] * self.L0

        Bn_a_star = Bx_a_star * nx + By_a_star * ny
        Bn_b_star = Bx_b_star * nx + By_b_star * ny
        loss_Bn = torch.mean(corner_w * ((Bn_a_star - Bn_b_star) ** 2))

        B_mag_a_phys = torch.sqrt(Bx_a_star**2 + By_a_star**2 + 1e-12) * (self.H0 / self.nu0)
        nu_a_star = domain_a.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_a_phys) / self.nu0
        hx_a, hy_a = domain_a.evaluate_magnetization_vector(pts)
        hx_a_star = hx_a / self.H0
        hy_a_star = hy_a / self.H0

        if domain_b is not None:
            B_mag_b_phys = torch.sqrt(Bx_b_star**2 + By_b_star**2 + 1e-12) * (self.H0 / self.nu0)
            nu_b_star = domain_b.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_b_phys) / self.nu0
            hx_b, hy_b = domain_b.evaluate_magnetization_vector(pts)
            hx_b_star = hx_b / self.H0
            hy_b_star = hy_b / self.H0
        else:
            nu_b_star = torch.ones_like(nu_a_star)
            hx_b_star = torch.zeros_like(hx_a_star)
            hy_b_star = torch.zeros_like(hy_a_star)

        Hx_a_star = nu_a_star * Bx_a_star - hx_a_star
        Hy_a_star = nu_a_star * By_a_star - hy_a_star
        Hx_b_star = nu_b_star * Bx_b_star - hx_b_star
        Hy_b_star = nu_b_star * By_b_star - hy_b_star

        Ht_a_star = Hx_a_star * tx + Hy_a_star * ty
        Ht_b_star = Hx_b_star * tx + Hy_b_star * ty
        nu_pair_scale = torch.clamp(torch.maximum(nu_a_star.detach(), nu_b_star.detach()), min=1e-6)
        ht_diff_normalized = (Ht_a_star - Ht_b_star) / nu_pair_scale
        loss_Ht = torch.mean(corner_w * (ht_diff_normalized ** 2))

        return loss_val, loss_Bn, loss_Ht

    def compute_loss_components(self, network_key, domain, surf_pts, background_key='background', contact_tol=1e-4):
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        bg_domain = sim_domain.background_domain
        other_domains = [
            d for d in sim_domain.domains_list
            if d is not domain and getattr(d, 'domain_name', '') != 'background'
        ]

        nx_a, ny_a, _, _, _ = self._compute_normals_and_tangents_from_outline(domain, surf_pts)

        remaining_mask = torch.ones(surf_pts.shape[0], dtype=torch.bool, device=surf_pts.device)
        total_val = torch.tensor(0.0, device=surf_pts.device)
        total_bn = torch.tensor(0.0, device=surf_pts.device)
        total_ht = torch.tensor(0.0, device=surf_pts.device)

        for other_dom in other_domains:
            if not remaining_mask.any():
                break
            other_sdf = other_dom.compute_signed_distance_field(surf_pts).view(-1)
            nx_b, ny_b, _, _, _ = self._compute_normals_and_tangents_from_outline(other_dom, surf_pts)
            normal_dot = (nx_a * nx_b + ny_a * ny_b).view(-1)
            contact_mask = remaining_mask & (torch.abs(other_sdf) <= contact_tol) & (normal_dot < -0.85)
            if contact_mask.any():
                other_key = f"domain_{other_dom.domain_id}"
                v_l, bn_l, ht_l = self._compute_pair_components(
                    network_key, domain, other_key, other_dom, surf_pts[contact_mask]
                )
                ratio = float(contact_mask.sum().item()) / float(surf_pts.shape[0])
                total_val = total_val + ratio * v_l
                total_bn = total_bn + ratio * bn_l
                total_ht = total_ht + ratio * ht_l
                remaining_mask = remaining_mask & (~contact_mask)

        if remaining_mask.any():
            v_bg, bn_bg, ht_bg = self._compute_pair_components(
                network_key, domain, background_key, bg_domain, surf_pts[remaining_mask]
            )
            ratio = float(remaining_mask.sum().item()) / float(surf_pts.shape[0])
            total_val = total_val + ratio * v_bg
            total_bn = total_bn + ratio * bn_bg
            total_ht = total_ht + ratio * ht_bg

        w_az = float(getattr(domain.training_config, 'interface_potential_weight', 1000.0))
        w_bn = float(getattr(domain.training_config, 'interface_bn_weight', 500.0))
        w_ht = float(getattr(domain.training_config, 'interface_ht_weight', 500.0))

        weighted_interface = w_az * total_val + w_bn * total_bn + w_ht * total_ht
        return weighted_interface, total_val, total_bn, total_ht

    def compute_loss(self, network_key, domain, surf_pts, background_key='background'):
        weighted_interface, _, _, _ = self.compute_loss_components(network_key, domain, surf_pts, background_key)
        return weighted_interface
'''

    # =========================================================================
    # 7. CẬP NHẬT: training_engine/training_manager.py
    # =========================================================================
    files_to_update[os.path.join(training_dir, 'training_manager.py')] = '''import time
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

        raw_pde_lin_sum = 0.0
        raw_pde_nonlin_sum = 0.0
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

                grad_A_star = torch.autograd.grad(outputs=A_z_star, inputs=pts, grad_outputs=torch.ones_like(A_z_star), create_graph=True)[0]
                Bx_star = grad_A_star[:, 1:2] * self.pde_evaluator.L0
                By_star = -grad_A_star[:, 0:1] * self.pde_evaluator.L0
                B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * (self.pde_evaluator.H0 / self.pde_evaluator.nu0)

                nu_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys).to(device)
                nu_lin_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)

                hx, hy = domain.evaluate_magnetization_vector(pts)
                hx, hy = hx.to(device), hy.to(device)
                jz = domain.evaluate_current_density(pts).to(device)
                
                res_lin, res_nonlin = self.pde_evaluator.compute_residual_components(
                    xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_linear_phys=nu_lin_phys
                )
                fading_w = self.pde_evaluator.compute_fading_weight(domain, pts)
                
                loss_lin = torch.mean((res_lin ** 2) * fading_w)
                loss_nonlin = torch.mean((res_nonlin ** 2) * fading_w)

                w_pde = float(domain.training_config.pde_weight)
                w_pde_nonlin = float(getattr(domain.training_config, 'pde_nonlinear_weight', 1.0))

                pde_domain_loss = w_pde * loss_lin + w_pde_nonlin * loss_nonlin
                total_loss = total_loss + pde_domain_loss

                raw_pde_lin_sum += loss_lin.item()
                raw_pde_nonlin_sum += loss_nonlin.item()
                weighted_pde_sum += pde_domain_loss.item()

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

        self.last_loss_breakdown = {
            "raw_pde_lin": raw_pde_lin_sum,
            "raw_pde_nonlin": raw_pde_nonlin_sum,
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
                    f"PDE(Lin/NonLin) = {bd.get('raw_pde_lin', 0.0):.3e}/{bd.get('raw_pde_nonlin', 0.0):.3e} | "
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
        last_loss = [0.0]
        start_time = time.time()
        base_step = self.completed_adam_epochs
        
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
                print(
                    f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {loss.item():.4e} | "
                    f"PDE(Lin/NonLin) = {bd.get('raw_pde_lin', 0.0):.3e}/{bd.get('raw_pde_nonlin', 0.0):.3e}"
                )
            return loss
            
        self.optimizer_lbfgs.step(closure)
        return domains_data
'''

    # =========================================================================
    # 8. CẬP NHẬT: utils/training_visualizer.py
    # Bổ sung đồ thị Động Permeability Distribution mu_r(B) (Layout 2x4)
    # =========================================================================
    files_to_update[os.path.join(utils_dir, 'training_visualizer.py')] = '''import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

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
            net_key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{d_id}'
            mask_bool = (mat_id == float(d_id))
            if mask_bool.any():
                Az_sub_star = model.forward_network(net_key, eval_points)
                grad_sub_star = torch.autograd.grad(
                    outputs=Az_sub_star,
                    inputs=eval_points,
                    grad_outputs=torch.ones_like(Az_sub_star),
                    create_graph=True
                )[0]
                Az_sub_phys = Az_sub_star * sim_domain.A0
                Bx_sub_phys = grad_sub_star[:, 1:2] * sim_domain.A0
                By_sub_phys = -grad_sub_star[:, 0:1] * sim_domain.A0

                B_mag_sub_phys = torch.sqrt(Bx_sub_phys**2 + By_sub_phys**2 + 1e-12)

                nu_dom = domain.evaluate_reluctivity(eval_points, B_magnitude_tensor=B_mag_sub_phys).to(computation_device)
                hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
                hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
                jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

                res_lin, res_nonlin = pde_evaluator.compute_residual_components(
                    eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom, domain.evaluate_reluctivity(eval_points, is_nonlinear=False).to(computation_device)
                )

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
                nu_phys_all = torch.where(mask_bool, nu_dom.detach(), nu_phys_all)
                residual_total = torch.where(mask_bool, torch.abs((res_lin + res_nonlin).detach()), residual_total)
        
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
        mu_r_numpy = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys_all, min=1e-6)).cpu().numpy().reshape(self.resolution, self.resolution)
        residual_numpy = residual_total.cpu().numpy().reshape(self.resolution, self.resolution)
        
        all_pts = []
        for key, d in domains_data.items():
            if d["interior_points"].shape[0] > 0:
                all_pts.append(d["interior_points"].detach().cpu().numpy())
            if d["surface_points"].shape[0] > 0:
                all_pts.append(d["surface_points"].detach().cpu().numpy())
        points_numpy = np.vstack(all_pts) if all_pts else np.empty((0, 2))
        
        # Grid layout 2x4 với biểu đồ mu_r động
        fig, axs = plt.subplots(2, 4, figsize=(22, 10))
        
        # [0, 0] Collocation Points
        if points_numpy.shape[0] > 0:
            axs[0, 0].scatter(points_numpy[:, 0], points_numpy[:, 1], s=1, c='black', alpha=0.5)
        axs[0, 0].set_title("Collocation Points")
        axs[0, 0].set_xlim(self.x_bounds)
        axs[0, 0].set_ylim(self.y_bounds)
        axs[0, 0].set_aspect('equal')
        
        # [0, 1] PDE Residual Error
        res_abs = np.clip(np.abs(residual_numpy), 1e-6, 1e2)
        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, res_abs, levels=50, cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=1e1))
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')

        # [0, 2] Dynamic Relative Permeability mu_r(B)
        contour_mur = axs[0, 2].contourf(self.X_grid, self.Y_grid, mu_r_numpy, levels=50, cmap="cividis")
        fig.colorbar(contour_mur, ax=axs[0, 2], label=r"Relative Permeability ($\mu_r$)")
        axs[0, 2].set_title(r"Permeability Distribution ($\mu_r$)")
        axs[0, 2].set_aspect('equal')
        
        # [0, 3] Loss Optimization
        axs[0, 3].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 3].set_yscale('log')
        axs[0, 3].set_title("Loss Optimization")
        axs[0, 3].set_xlabel("Step / Epoch")
        axs[0, 3].set_ylabel("Loss")
        axs[0, 3].grid(True, which="both", ls="-", alpha=0.2)
        
        # [1, 0] Az
        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=50, cmap="jet")
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        # [1, 1] |B|
        b_max_val = max(float(np.max(B_mag_numpy)), 1e-3)
        levels_b = np.linspace(0.0, b_max_val, 50)
        contour_b = axs[1, 1].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=levels_b, cmap="jet", vmin=0.0, vmax=b_max_val)
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title("Flux Density Magnitude ($|B|$)")
        axs[1, 1].set_aspect('equal')

        # [1, 2] Flux Density Vectors
        contour_b_bg = axs[1, 2].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=levels_b, cmap="jet", alpha=0.3, vmin=0.0, vmax=b_max_val)
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

        # [1, 3] Material Domain Layout
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
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps, loop=0)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")

    def generate_mp4(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        codec = getattr(cfg, 'mp4_codec', 'libx264')
        quality = getattr(cfg, 'mp4_quality', 8)

        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo MP4.")
            return

        valid_frames = [f for f in self.frame_paths if os.path.exists(f)]
        if not valid_frames:
            print("[!] CẢNH BÁO: Tất cả frame paths đều không tồn tại trên đĩa.")
            return

        full_mp4_path = os.path.join(self.output_directory, output_filename)
        print(f"[*] Đang xuất Video MP4 ({len(valid_frames)} frames, {fps} fps)...")
        
        exported = False
        try:
            writer = imageio.get_writer(
                full_mp4_path,
                fps=fps,
                codec=codec,
                pixelformat='yuv420p',
                quality=quality
            )
            for f in valid_frames:
                writer.append_data(imageio.imread(f))
            writer.close()
            exported = True
            print(f"[*] Đã xuất MP4 Video thành công (imageio) tại: {full_mp4_path}")
        except Exception as exc:
            print(f"[!] Không thể xuất MP4 bằng imageio-ffmpeg ({exc}). Chuyển sang OpenCV fallback...")

        if not exported:
            try:
                import cv2
                sample = cv2.imread(valid_frames[0])
                height, width, _ = sample.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(full_mp4_path, fourcc, float(fps), (width, height))
                for f in valid_frames:
                    img = cv2.imread(f)
                    video_writer.write(img)
                video_writer.release()
                print(f"[*] Đã xuất MP4 Video thành công (OpenCV) tại: {full_mp4_path}")
            except Exception as cv_exc:
                print(f"[!] Lỗi khi xuất video MP4 bằng OpenCV: {cv_exc}")

    def generate_animations(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        gen_gif = getattr(cfg, 'generate_gif_animation', True)
        gen_mp4 = getattr(cfg, 'generate_mp4_video', True)

        if gen_gif:
            self.generate_gif()
        if gen_mp4:
            self.generate_mp4()
'''

    # =========================================================================
    # 9. CẬP NHẬT: electro_magnetic_pinn.py
    # Quy trình Huấn luyện 3 Giai đoạn
    # =========================================================================
    files_to_update[pinn_main_file] = '''import torch
import torch.nn as nn
from types import SimpleNamespace
from architectures.xpinn_architecture import XPINNArchitecture
from training_engine.training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from physics_domain.physical_equations.interface_loss import InterfaceLoss
from simulation_domain_engine.simulation_domain import SimulationDomain
from physics_domain.collocation_sampler import CollocationSampler
from utils.training_visualizer import TrainingVisualizer

class ElectroMagneticPINN:
    def __init__(self):
        self.sampler_config = SimpleNamespace(
            dynamic_resampling=True,
            resample_interval=100,
            resample_before_lbfgs=True
        )
        
        self.training_config = SimpleNamespace(
            learning_rate_adam=1e-3,
            target_loss=1e-5,
            epochs_adam=2500,
            epochs_lbfgs=800,
            lbfgs_learning_rate=0.8,
            lbfgs_maximum_iterations=800,
            lbfgs_maximum_evaluations=1000,
            lbfgs_tolerance_gradient=1e-8,
            lbfgs_tolerance_change=1e-10,
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
            mp4_quality=8
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

    def execute_training_process(self):
        if self.pinn_architecture_instance is None:
            self._build_system()

        def resample_data_fn():
            return self.collocation_sampler_instance.generate_all_domain_points()

        domains_data = resample_data_fn()
        sim_domain = self.simulation_domain_engine_instance

        total_adam_epochs = self.training_config.epochs_adam
        epochs_stage_1a = int(total_adam_epochs * 0.3)
        epochs_stage_1b = total_adam_epochs - epochs_stage_1a

        # STAGE 1A: ADAM - LINEAR WARM-UP
        print(f"\\n--- [STAGE 1A] Adam Linear Warm-up ({epochs_stage_1a} Epochs) ---")
        for domain in sim_domain.domains_list:
            domain.physics_config.is_nonlinear = False

        domains_data = self.training_manager_instance.train_adam(
            epochs=epochs_stage_1a, 
            domains_data=domains_data,
            dynamic_resampling=self.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn,
            resample_interval=self.sampler_config.resample_interval
        )

        # STAGE 1B: ADAM - NON-LINEAR ADAPTATION
        print(f"\\n--- [STAGE 1B] Adam Non-linear Adaptation ({epochs_stage_1b} Epochs) ---")
        for domain in sim_domain.domains_list:
            if getattr(domain.physics_config, 'bh_curve_data', None) is not None:
                domain.physics_config.is_nonlinear = True

        domains_data = self.training_manager_instance.train_adam(
            epochs=epochs_stage_1b, 
            domains_data=domains_data,
            dynamic_resampling=self.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn,
            resample_interval=self.sampler_config.resample_interval
        )

        # STAGE 2: L-BFGS REFINEMENT
        if getattr(self.sampler_config, 'resample_before_lbfgs', True):
            domains_data = resample_data_fn()

        lbfgs_iters = int(self.training_config.epochs_lbfgs)
        print(f"\\n--- [STAGE 2] L-BFGS Refinement (Max {lbfgs_iters} Iterations) ---")
        self.training_manager_instance.train_lbfgs(
            max_iterations=lbfgs_iters, 
            domains_data=domains_data
        )
        
        if self.visualizer_instance:
            self.visualizer_instance.generate_animations()

    def evaluate_fields(self, points_tensor):
        self.pinn_architecture_instance.eval()
        points_tensor = points_tensor.to(self.computation_device).detach().clone().requires_grad_(True)
        
        props = self.simulation_domain_engine_instance.evaluate_global_physical_properties(points_tensor)
        mat_id = props["material_classification"]
        
        A_z_phys = torch.zeros((points_tensor.shape[0], 1), device=self.computation_device)
        B_x_phys = torch.zeros((points_tensor.shape[0], 1), device=self.computation_device)
        B_y_phys = torch.zeros((points_tensor.shape[0], 1), device=self.computation_device)
        
        for domain in self.simulation_domain_engine_instance.domains_list:
            d_id = domain.domain_id
            if getattr(domain, 'domain_name', '') == 'background':
                net_key = 'background'
            else:
                net_key = f'domain_{d_id}'
                
            mask_bool = (mat_id == float(d_id))
            if mask_bool.any():
                Az_sub_star = self.pinn_architecture_instance.forward_network(net_key, points_tensor)
                grad_sub_star = torch.autograd.grad(
                    outputs=Az_sub_star,
                    inputs=points_tensor,
                    grad_outputs=torch.ones_like(Az_sub_star),
                    create_graph=False
                )[0]
                Az_sub_phys = Az_sub_star * self.A0
                Bx_sub_phys = grad_sub_star[:, 1:2] * self.A0
                By_sub_phys = -grad_sub_star[:, 0:1] * self.A0

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), By_sub_phys)
        
        return A_z_phys.detach(), B_x_phys.detach(), B_y_phys.detach()
'''

    # =========================================================================
    # 10. CẬP NHẬT: tests/test_simulation4.py (Modern Test với B-H steel_1008)
    # =========================================================================
    files_to_update[os.path.join(tests_dir, 'test_simulation4.py')] = '''import os
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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from electro_magnetic_pinn import ElectroMagneticPINN
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY


def run_modern_simulation_test():
    output_dir = "figure_modern_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN Mô hình Hiện đại (Modern API Non-linear Test)...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_before_lbfgs = True

    model.training_config.epochs_adam = 3000
    model.training_config.epochs_lbfgs = 1500
    model.training_config.learning_rate_adam = 1e-3
    model.training_config.target_loss = 1e-7

    model.visualization_config.active = True
    model.visualization_config.update_interval = 50
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "modern_training.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "modern_training.mp4"

    def apply_standard_config(domain, int_pts=2000, surf_pts=1200, pde_nonlin_w=2.0):
        domain.training_config.pde_weight = 1.0
        domain.training_config.pde_nonlinear_weight = pde_nonlin_w
        domain.training_config.interface_weight = 1.0
        domain.training_config.interface_potential_weight = 1000.0
        domain.training_config.interface_bn_weight = 400.0
        domain.training_config.interface_ht_weight = 400.0
        domain.training_config.pde_fading_beta = 80.0
        domain.training_config.number_of_interior_points = int_pts
        domain.training_config.number_of_surface_points = surf_pts
        
        domain.network_config.hidden_layers = 4
        domain.network_config.hidden_neurons = 64
        domain.network_config.activation_function = nn.SiLU()
        domain.network_config.fourier_features = True
        domain.network_config.fourier_sigma = 1.2
        domain.network_config.fourier_mapping_size = 128
        return domain.build_domain()

    # 1. Khung Ách sắt Phi tuyến (Iron Yoke - steel_1008)
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
    apply_standard_config(iron_yoke, int_pts=3000, surf_pts=1600, pde_nonlin_w=2.0)

    # 2. Nam châm trái (N30UH)
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
    apply_standard_config(magnet_left)

    # 3. Nam châm phải (N30UH)
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
    apply_standard_config(magnet_right)

    # 4. Background Air
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        pde_nonlinear_weight=1.0,
        interface_weight=1.0,
        pde_fading_beta=80.0,
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
    sim_domain.plot_problem_definition(resolution=100)
    
    print("[*] Tiến trình huấn luyện 3 Giai đoạn (Adam Linear -> Adam Non-linear -> L-BFGS)...")
    model.execute_training_process()

    print("[*] Đang đánh giá trường từ trường dự đoán...")
    resolution = 180
    xb, yb = sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    eval_pts = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    A_z_pred, B_x_pred, B_y_pred = model.evaluate_fields(eval_pts)
    
    assert torch.all(torch.isfinite(A_z_pred)) and torch.all(torch.isfinite(B_x_pred)), "Lỗi: Trường dự đoán chứa NaN/Inf!"

    A_z_grid = A_z_pred.cpu().numpy().reshape(resolution, resolution)
    B_x_grid = B_x_pred.cpu().numpy().reshape(resolution, resolution)
    B_y_grid = B_y_pred.cpu().numpy().reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)

    # Đánh giá độ từ thẩm mu_r động tại kết quả cuối
    B_mag_phys_tensor = torch.tensor(B_mag_grid.ravel(), dtype=torch.float32).unsqueeze(1)
    nu_pred = iron_yoke.evaluate_reluctivity(eval_pts, B_magnitude_tensor=B_mag_phys_tensor, is_nonlinear=True)
    mu_r_grid = (VACUUM_RELUCTIVITY / torch.clamp(nu_pred, min=1e-6)).cpu().numpy().reshape(resolution, resolution)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    c0 = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=60, cmap="jet")
    fig.colorbar(c0, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
    axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
    axs[0, 0].set_aspect('equal')

    b_max = max(float(np.max(B_mag_grid)), 1e-3)
    c1 = axs[0, 1].contourf(X_grid, Y_grid, B_mag_grid, levels=np.linspace(0, b_max, 60), cmap="jet")
    fig.colorbar(c1, ax=axs[0, 1], label=r"$|B|$ (T)")
    axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_aspect('equal')

    c2 = axs[1, 0].contourf(X_grid, Y_grid, mu_r_grid, levels=60, cmap="cividis")
    fig.colorbar(c2, ax=axs[1, 0], label=r"$\mu_r$")
    axs[1, 0].set_title(r"Final Permeability Distribution ($\mu_r$)")
    axs[1, 0].set_aspect('equal')

    c3 = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=60, cmap="coolwarm")
    fig.colorbar(c3, ax=axs[1, 1], label=r"$B_y$ (T)")
    axs[1, 1].set_title(r"Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/modern_final_fields.png", dpi=150)
    plt.close('all')

    print(f"[✔] HOÀN THÀNH BÀI KIỂM THỬ MÔ HÌNH PHI TUYẾN 3 GIAI ĐOẠN! Kết quả đã được lưu tại {output_dir}/")

if __name__ == '__main__':
    run_modern_simulation_test()
'''

    # Ghi lại toàn bộ các file đã cập nhật
    for file_path, content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + '\n')
        print(f"[*] Đã tạo/cập nhật thành công: {os.path.relpath(file_path, project_root)}")

    print("\n[✔] TẢI LẠI HỆ THỐNG HOÀN TẤT: Đã cập nhật đồ thị động Permeability Distribution mu_r(B) vào Visualizer và sẵn sàng kịch bản test_simulation4.py!")

if __name__ == '__main__':
    apply_full_optimizations_and_visuals()