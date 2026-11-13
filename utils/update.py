import os
from pathlib import Path


def resolve_project_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src" / "physics_informed_model").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing 'src/physics_informed_model'.")


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"[UPDATED] {path}")


RELOBRALO_BALANCER_CODE = r"""import math
import random


class ReLoBRaLoBalancer:
    def __init__(
        self,
        keys=("loss_pde", "loss_a", "loss_bn", "loss_ht"),
        alpha=0.999,
        temperature=0.1,
        rho=0.999,
        eps=1e-12
    ):
        self.keys = tuple(keys)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.rho = float(rho)
        self.eps = float(eps)
        self.lambdas = {k: 1.0 for k in self.keys}
        self.loss_init = None
        self.loss_prev = None
        self.step_count = 0

    def _to_scalar(self, val):
        if hasattr(val, "detach"):
            return float(val.detach().item())
        return float(val)

    def _scaled_softmax(self, ratios):
        k_count = len(ratios)
        if k_count == 0:
            return []
        if k_count == 1:
            return [1.0]
        tau = max(self.temperature, self.eps)
        scaled = [max(-50.0, min(50.0, r / tau)) for r in ratios]
        max_val = max(scaled)
        exps = [math.exp(s - max_val) for s in scaled]
        sum_exp = max(sum(exps), self.eps)
        return [float(k_count) * (e / sum_exp) for e in exps]

    def reset_anchors(self, reset_weights=False):
        self.loss_init = None
        self.loss_prev = None
        if reset_weights:
            self.lambdas = {k: 1.0 for k in self.keys}

    def get_weights(self, keys=None):
        target_keys = tuple(keys) if keys is not None else self.keys
        return {k: float(self.lambdas.get(k, 1.0)) for k in target_keys}

    def update(self, losses_dict, is_resampling_step=False):
        current_keys = tuple(losses_dict.keys())
        for k in current_keys:
            if k not in self.lambdas:
                self.lambdas[k] = 1.0

        current_vals = {k: self._to_scalar(losses_dict[k]) for k in current_keys}

        for v in current_vals.values():
            if not math.isfinite(v):
                return self.get_weights(current_keys)

        if self.loss_init is None or self.loss_prev is None:
            self.loss_init = dict(current_vals)
            self.loss_prev = dict(current_vals)
            self.step_count = 0
            return self.get_weights(current_keys)

        for k, v in current_vals.items():
            if k not in self.loss_init or self.loss_init[k] <= self.eps:
                self.loss_init[k] = v
            if k not in self.loss_prev or self.loss_prev[k] <= self.eps:
                self.loss_prev[k] = v

        if is_resampling_step:
            self.loss_prev = dict(current_vals)
            return self.get_weights(current_keys)

        active_keys = [k for k in current_keys if current_vals[k] > self.eps]
        if len(active_keys) <= 1:
            for k in current_keys:
                self.lambdas[k] = 1.0
            self.loss_prev = dict(current_vals)
            self.step_count += 1
            return self.get_weights(current_keys)

        short_ratios = [
            current_vals[k] / max(self.loss_prev[k], self.eps)
            for k in active_keys
        ]
        long_ratios = [
            current_vals[k] / max(self.loss_init[k], self.eps)
            for k in active_keys
        ]

        hat_short = self._scaled_softmax(short_ratios)
        hat_long = self._scaled_softmax(long_ratios)

        rho_sample = 1.0 if random.random() < self.rho else 0.0

        for idx, k in enumerate(active_keys):
            prev_lambda = float(self.lambdas.get(k, 1.0))
            hist_lambda = rho_sample * prev_lambda + (1.0 - rho_sample) * hat_long[idx]
            new_lambda = self.alpha * hist_lambda + (1.0 - self.alpha) * hat_short[idx]
            self.lambdas[k] = float(new_lambda)

        for k in current_keys:
            if k not in active_keys:
                self.lambdas[k] = 1.0

        self.loss_prev = dict(current_vals)
        self.step_count += 1
        return self.get_weights(current_keys)

    def state_dict(self):
        return {
            "keys": list(self.keys),
            "alpha": self.alpha,
            "temperature": self.temperature,
            "rho": self.rho,
            "eps": self.eps,
            "lambdas": dict(self.lambdas),
            "loss_init": dict(self.loss_init) if self.loss_init is not None else None,
            "loss_prev": dict(self.loss_prev) if self.loss_prev is not None else None,
            "step_count": int(self.step_count)
        }

    def load_state_dict(self, state):
        if not state:
            return
        if "keys" in state:
            self.keys = tuple(state["keys"])
        self.alpha = float(state.get("alpha", self.alpha))
        self.temperature = float(state.get("temperature", self.temperature))
        self.rho = float(state.get("rho", self.rho))
        self.eps = float(state.get("eps", self.eps))
        if "lambdas" in state and state["lambdas"] is not None:
            self.lambdas = {k: float(v) for k, v in state["lambdas"].items()}
        self.loss_init = (
            {k: float(v) for k, v in state["loss_init"].items()}
            if state.get("loss_init") is not None else None
        )
        self.loss_prev = (
            {k: float(v) for k, v in state["loss_prev"].items()}
            if state.get("loss_prev") is not None else None
        )
        self.step_count = int(state.get("step_count", 0))
"""


MATERIAL_DOMAIN_CODE = r"""from typing import Tuple, Union, List, Optional, Any
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
            pde_fading_beta=100.0,
            sampling_density_multiplier=1.0,
            number_of_interior_points=2000,
            number_of_surface_points=1200
        )
        
        self.network_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU(),
            fourier_features=True,
            fourier_sigma=1.2,
            fourier_mapping_size=128
        )

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
        return self.build_domain()

    def _compute_constant_bh_curve(self) -> Any:
        return _compute_constant_bh_curve_module.compute_constant_bh_curve(self)

    def _compute_section_area(self) -> float:
        return _compute_section_area_module.compute_section_area(self)

    def _compute_current_density(self) -> Union[float, Any]:
        return _compute_current_density_module.compute_current_density(self)

    def compute_signed_distance_field(self, points_tensor: torch.Tensor) -> torch.Tensor:
        if self.outline_tensor is None:
            raise RuntimeError(f"Domain '{self.domain_name}' has not called build_domain().")
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
"""


INTERFACE_LOSS_CODE = r"""import torch


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

    def compute_pointwise_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            empty = torch.empty((0, 1), device=pts.device, dtype=pts.dtype)
            return empty, empty, empty

        model = self.electro_magnetic_pinn.pinn_architecture_instance
        nx, ny, tx, ty, corner_w = self._compute_normals_and_tangents_from_outline(domain_a, pts)

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            _, _, _, _, corner_w_b = self._compute_normals_and_tangents_from_outline(domain_b, pts)
            corner_w = torch.minimum(corner_w, corner_w_b)

        Az_a = model.forward_network(key_a, pts)
        Az_b = model.forward_network(key_b, pts)

        err_val = (Az_a - Az_b) ** 2

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
        err_Bn = corner_w * ((Bn_a_star - Bn_b_star) ** 2)

        if hasattr(model, 'nu_networks') and key_a in model.nu_networks:
            nu_a_star = model.forward_nu_network(key_a, pts)
        else:
            B_mag_a_phys = torch.sqrt(Bx_a_star**2 + By_a_star**2 + 1e-12) * (self.H0 / self.nu0)
            nu_a_star = domain_a.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_a_phys, is_nonlinear=True) / self.nu0

        hx_a, hy_a = domain_a.evaluate_magnetization_vector(pts)
        hx_a_star = hx_a / self.H0
        hy_a_star = hy_a / self.H0

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            if hasattr(model, 'nu_networks') and key_b in model.nu_networks:
                nu_b_star = model.forward_nu_network(key_b, pts)
            else:
                B_mag_b_phys = torch.sqrt(Bx_b_star**2 + By_b_star**2 + 1e-12) * (self.H0 / self.nu0)
                nu_b_star = domain_b.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_b_phys, is_nonlinear=True) / self.nu0
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
        err_Ht = corner_w * (ht_diff_normalized ** 2)

        return err_val, err_Bn, err_Ht

    def _compute_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            zero = torch.tensor(0.0, device=pts.device)
            return zero, zero, zero, zero

        err_val, err_Bn, err_Ht = self.compute_pointwise_pair_components(
            key_a, domain_a, key_b, domain_b, pts
        )

        loss_val = torch.mean(err_val)
        loss_Bn = torch.mean(err_Bn)
        loss_Ht = torch.mean(err_Ht)

        w_az = float(getattr(domain_a.training_config, 'interface_potential_weight', 1.0))
        w_bn = float(getattr(domain_a.training_config, 'interface_bn_weight', 1.0))
        w_ht = float(getattr(domain_a.training_config, 'interface_ht_weight', 1.0))

        weighted_interface = w_az * loss_val + w_bn * loss_Bn + w_ht * loss_Ht
        return weighted_interface, loss_val, loss_Bn, loss_Ht

    def compute_loss_components(self, network_key, domain, surf_pts, background_key='background'):
        bg_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance.background_domain
        return self._compute_pair_components(network_key, domain, background_key, bg_domain, surf_pts)

    def compute_loss(self, network_key, domain, surf_pts, background_key='background'):
        weighted_interface, _, _, _ = self.compute_loss_components(network_key, domain, surf_pts, background_key)
        return weighted_interface
"""


GLOBAL_PHYSICAL_PROPERTIES_CODE = r"""import torch

VACUUM_RELUCTIVITY = 795774.715459


def evaluate_global_physical_properties(domains_list, points_tensor):
    number_of_points = points_tensor.shape[0]
    computation_device = points_tensor.device
    computation_dtype = points_tensor.dtype

    background_domain = None
    material_domains = []
    for domain in domains_list:
        if getattr(domain, 'domain_name', '') == 'background':
            background_domain = domain
        else:
            material_domains.append(domain)

    if background_domain is not None:
        global_reluctivity = background_domain.evaluate_reluctivity(points_tensor).clone()
        base_hx, base_hy = background_domain.evaluate_magnetization_vector(points_tensor)
        global_hx = base_hx.clone()
        global_hy = base_hy.clone()
        global_jz = background_domain.evaluate_current_density(points_tensor).clone()
        base_pde_weight = float(getattr(background_domain.training_config, 'pde_weight', 1.0))
        base_interface_weight = float(getattr(background_domain.training_config, 'interface_weight', 1.0))
        base_sampling_density = float(getattr(background_domain.training_config, 'sampling_density_multiplier', 1.0))
        base_mat_id = float(background_domain.domain_id) if background_domain.domain_id is not None else float(len(domains_list))
    else:
        global_reluctivity = torch.full((number_of_points, 1), VACUUM_RELUCTIVITY, dtype=computation_dtype, device=computation_device)
        global_hx = torch.zeros((number_of_points, 1), dtype=computation_dtype, device=computation_device)
        global_hy = torch.zeros((number_of_points, 1), dtype=computation_dtype, device=computation_device)
        global_jz = torch.zeros((number_of_points, 1), dtype=computation_dtype, device=computation_device)
        base_pde_weight = 1.0
        base_interface_weight = 1.0
        base_sampling_density = 1.0
        base_mat_id = 0.0

    global_mat_id = torch.full((number_of_points, 1), base_mat_id, dtype=computation_dtype, device=computation_device)
    global_pde_weight = torch.full((number_of_points, 1), base_pde_weight, dtype=computation_dtype, device=computation_device)
    global_interface_weight = torch.full((number_of_points, 1), base_interface_weight, dtype=computation_dtype, device=computation_device)
    global_sampling_density = torch.full((number_of_points, 1), base_sampling_density, dtype=computation_dtype, device=computation_device)

    if not material_domains:
        return {
            "reluctivity": global_reluctivity,
            "coercive_field_x": global_hx,
            "coercive_field_y": global_hy,
            "current_density_z": global_jz,
            "material_classification": global_mat_id,
            "pde_weight": global_pde_weight,
            "interface_weight": global_interface_weight,
            "sampling_density_multiplier": global_sampling_density
        }

    best_sdf = torch.full((number_of_points, 1), 1e9, dtype=computation_dtype, device=computation_device)

    for idx, domain_object in enumerate(material_domains):
        sdf = domain_object.compute_signed_distance_field(points_tensor).view(-1, 1)
        is_inside_and_closer = (sdf <= 0.0) & (sdf < best_sdf)
        best_sdf = torch.where(is_inside_and_closer, sdf, best_sdf)

        nu_dom = domain_object.evaluate_reluctivity(points_tensor)
        hx_dom, hy_dom = domain_object.evaluate_magnetization_vector(points_tensor)
        jz_dom = domain_object.evaluate_current_density(points_tensor)
        domain_id_val = float(domain_object.domain_id) if domain_object.domain_id is not None else float(idx + 1)

        dom_pde_w = float(getattr(domain_object.training_config, 'pde_weight', 1.0))
        dom_if_w = float(getattr(domain_object.training_config, 'interface_weight', 1.0))
        dom_samp_mult = float(getattr(domain_object.training_config, 'sampling_density_multiplier', 1.0))

        global_reluctivity = torch.where(is_inside_and_closer, nu_dom, global_reluctivity)
        global_hx = torch.where(is_inside_and_closer, hx_dom, global_hx)
        global_hy = torch.where(is_inside_and_closer, hy_dom, global_hy)
        global_jz = torch.where(is_inside_and_closer, jz_dom, global_jz)
        global_mat_id = torch.where(is_inside_and_closer, torch.full_like(global_mat_id, domain_id_val), global_mat_id)
        global_pde_weight = torch.where(is_inside_and_closer, torch.full_like(global_pde_weight, dom_pde_w), global_pde_weight)
        global_interface_weight = torch.where(is_inside_and_closer, torch.full_like(global_interface_weight, dom_if_w), global_interface_weight)
        global_sampling_density = torch.where(is_inside_and_closer, torch.full_like(global_sampling_density, dom_samp_mult), global_sampling_density)

    return {
        "reluctivity": global_reluctivity,
        "coercive_field_x": global_hx,
        "coercive_field_y": global_hy,
        "current_density_z": global_jz,
        "material_classification": global_mat_id,
        "pde_weight": global_pde_weight,
        "interface_weight": global_interface_weight,
        "sampling_density_multiplier": global_sampling_density
    }
"""


ELECTRO_MAGNETIC_PINN_CODE = r"""import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from types import SimpleNamespace
from simulation_domain_engine.simulation_domain import SimulationDomain
from training_engine.resolve_domain_net_key import resolve_domain_net_key

from electro_magnetic_pinn_engine.build_system import _build_system
from electro_magnetic_pinn_engine.evaluate_fields import evaluate_fields
from electro_magnetic_pinn_engine.plot_fields import plot_fields
from electro_magnetic_pinn_engine.execute_training_process import execute_training_process

from checkpoint_engine.save_checkpoint import save_checkpoint
from checkpoint_engine.load_checkpoint import load_checkpoint


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
            enable_lbfgs_after_warmup=False,
            enable_lbfgs_after_fpi_step=False,
            linear_warmup_epochs_ratio=0.3,

            enable_relobralo=False,
            relobralo_alpha=0.999,
            relobralo_temperature=0.1,
            relobralo_rho=0.999,

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
            lbfgs_warmup_iterations=1500,
            lbfgs_fpi_step_iterations=1500,
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
            display_colab_frames=True,

            contour_levels=60,

            res_scale="log",          
            res_vmin=1e-4,            
            res_vmax=10.0,             
            res_cmap="inferno",                                   
            res_discrete_levels=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], 

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
            b_mag_cmap="ansys_fem",
            b_mag_unit="Tesla",             
            b_mag_number_of_division=100,   

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
        return _build_system(self)

    def evaluate_fields(self, points_tensor, return_numpy=False):
        return evaluate_fields(self, points_tensor, return_numpy=return_numpy)

    def plot_fields(self, resolution=180, save_path=None, show=False, figsize=(12, 10)):
        return plot_fields(self, resolution=resolution, save_path=save_path, show=show, figsize=figsize)

    def execute_training_process(self):
        return execute_training_process(self)

    def save_checkpoint(self, filepath=None):
        return save_checkpoint(self, filepath=filepath)

    def load_checkpoint(self, filepath=None):
        return load_checkpoint(self, filepath=filepath)

    def visualize_collocation_sampling(self, domains_data=None, save_path=None, show=True):
        if domains_data is None:
            if self.collocation_sampler_instance is None:
                from physics_domain.collocation_sampler import CollocationSampler
                self.collocation_sampler_instance = CollocationSampler(
                    electro_magnetic_pinn=self,
                    device=getattr(self, 'computation_device', torch.device('cpu'))
                )
            domains_data = self.collocation_sampler_instance.generate_all_domain_points()

        from visualization.plot_collocation_sampling import plot_collocation_sampling
        return plot_collocation_sampling(domains_data=domains_data, save_path=save_path, show=show)

    def display_training_points_count(self, domains_data=None):
        if self.collocation_sampler_instance is None:
            from physics_domain.collocation_sampler import CollocationSampler
            self.collocation_sampler_instance = CollocationSampler(
                electro_magnetic_pinn=self,
                device=getattr(self, 'computation_device', torch.device('cpu'))
            )
        return self.collocation_sampler_instance.get_training_points_count(domains_data=domains_data)
"""


TRAINING_ENGINE_INIT_CODE = r"""from .training_manager import TrainingManager
from .resolve_domain_net_key import resolve_domain_net_key
from .pretrain_nu_networks import pretrain_nu_networks
from .fit_nu_network_step import fit_nu_network_step
from .compute_loss import compute_loss
from .train_adam import train_adam
from .train_lbfgs import train_lbfgs
from .relobralo_balancer import ReLoBRaLoBalancer

__all__ = [
    "TrainingManager",
    "resolve_domain_net_key",
    "pretrain_nu_networks",
    "fit_nu_network_step",
    "compute_loss",
    "train_adam",
    "train_lbfgs",
    "ReLoBRaLoBalancer"
]
"""


TRAINING_MANAGER_CODE = r"""import time
import torch
import torch.optim as optim

from training_engine.resolve_domain_net_key import resolve_domain_net_key
from training_engine.pretrain_nu_networks import pretrain_nu_networks
from training_engine.fit_nu_network_step import fit_nu_network_step
from training_engine.compute_loss import compute_loss
from training_engine.train_adam import train_adam
from training_engine.train_lbfgs import train_lbfgs
from training_engine.relobralo_balancer import ReLoBRaLoBalancer


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

        self.enable_relobralo = bool(getattr(t_cfg, 'enable_relobralo', False))
        self.relobralo_balancer = None
        if self.enable_relobralo:
            self.relobralo_balancer = ReLoBRaLoBalancer(
                keys=("loss_pde", "loss_a", "loss_bn", "loss_ht"),
                alpha=float(getattr(t_cfg, 'relobralo_alpha', 0.999)),
                temperature=float(getattr(t_cfg, 'relobralo_temperature', 0.1)),
                rho=float(getattr(t_cfg, 'relobralo_rho', 0.999))
            )

        self.last_loss_breakdown = {}
        self.completed_adam_epochs = 0
        self.global_step = 0

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
        return pretrain_nu_networks(self, domains_data, epochs=epochs, lr=lr)

    def fit_nu_network_step(self, domains_data, alpha_damping=None, epochs=None, lr=None):
        return fit_nu_network_step(self, domains_data, alpha_damping=alpha_damping, epochs=epochs, lr=lr)

    def compute_loss(self, domains_data, update_relobralo=True, is_resampling_step=False):
        return compute_loss(
            self,
            domains_data,
            update_relobralo=update_relobralo,
            is_resampling_step=is_resampling_step
        )

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        return train_adam(self, epochs, domains_data, dynamic_resampling=dynamic_resampling, resample_fn=resample_fn, resample_interval=resample_interval)

    def train_lbfgs(self, max_iterations, domains_data):
        return train_lbfgs(self, max_iterations, domains_data)
"""


COMPUTE_LOSS_CODE = r"""import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key


def compute_loss(training_manager_instance, domains_data, update_relobralo=True, is_resampling_step=False):
    model = training_manager_instance.model
    device = next(model.parameters()).device
    dummy_param = next(model.parameters())
    zero_tensor = (dummy_param * 0.0).sum()

    total_pde_tensor = zero_tensor
    total_az_tensor = zero_tensor
    total_bn_tensor = zero_tensor
    total_ht_tensor = zero_tensor

    az_if_sum = 0.0
    bn_if_sum = 0.0
    ht_if_sum = 0.0
    weighted_if_sum = 0.0

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    pinn = training_manager_instance.electro_magnetic_pinn
    bg_domain = pinn.simulation_domain_engine_instance.background_domain

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
            w_pde = float(getattr(domain.training_config, 'pde_weight', 1.0))

            scaled_pde_loss = w_pde * loss_pde_domain
            total_pde_tensor = total_pde_tensor + scaled_pde_loss

        outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
        if outer_pts.shape[0] > 0 and net_key != 'background':
            if not outer_pts.requires_grad:
                outer_pts = outer_pts.detach().clone().requires_grad_(True)
                
            _, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
            )
            w_if = float(getattr(domain.training_config, 'interface_weight', 1.0))
            w_az = float(getattr(domain.training_config, 'interface_potential_weight', 1.0))
            w_bn = float(getattr(domain.training_config, 'interface_bn_weight', 1.0))
            w_ht = float(getattr(domain.training_config, 'interface_ht_weight', 1.0))

            total_az_tensor = total_az_tensor + (w_if * w_az * val_l)
            total_bn_tensor = total_bn_tensor + (w_if * w_bn * bn_l)
            total_ht_tensor = total_ht_tensor + (w_if * w_ht * ht_l)

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += (w_if * (w_az * val_l + w_bn * bn_l + w_ht * ht_l)).item()

    for pair_key, pair_data in shared_dict.items():
        pts = pair_data["interface_points"].to(device)
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)

            _, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts
            )
            dom_a = pair_data["domain_a"]
            w_if = float(getattr(dom_a.training_config, 'interface_weight', 1.0))
            w_az = float(getattr(dom_a.training_config, 'interface_potential_weight', 1.0))
            w_bn = float(getattr(dom_a.training_config, 'interface_bn_weight', 1.0))
            w_ht = float(getattr(dom_a.training_config, 'interface_ht_weight', 1.0))

            total_az_tensor = total_az_tensor + (w_if * w_az * val_l)
            total_bn_tensor = total_bn_tensor + (w_if * w_bn * bn_l)
            total_ht_tensor = total_ht_tensor + (w_if * w_ht * ht_l)

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += (w_if * (w_az * val_l + w_bn * bn_l + w_ht * ht_l)).item()

    balancer = getattr(training_manager_instance, 'relobralo_balancer', None)
    if balancer is not None:
        raw_losses = {
            "loss_pde": total_pde_tensor,
            "loss_a": total_az_tensor,
            "loss_bn": total_bn_tensor,
            "loss_ht": total_ht_tensor
        }
        if update_relobralo:
            lambdas = balancer.update(raw_losses, is_resampling_step=is_resampling_step)
        else:
            lambdas = balancer.get_weights(raw_losses.keys())

        total_loss = (
            lambdas["loss_pde"] * total_pde_tensor
            + lambdas["loss_a"] * total_az_tensor
            + lambdas["loss_bn"] * total_bn_tensor
            + lambdas["loss_ht"] * total_ht_tensor
        )
    else:
        lambdas = {
            "loss_pde": 1.0,
            "loss_a": 1.0,
            "loss_bn": 1.0,
            "loss_ht": 1.0
        }
        total_loss = total_pde_tensor + total_az_tensor + total_bn_tensor + total_ht_tensor

    training_manager_instance.last_loss_breakdown = {
        "pde_loss": total_pde_tensor.item(),
        "interface_Az": az_if_sum,
        "interface_Bn": bn_if_sum,
        "interface_Ht": ht_if_sum,
        "weighted_interface": weighted_if_sum,
        "relobralo_lambdas": dict(lambdas),
        "total": total_loss.item()
    }

    return total_loss
"""


TRAIN_ADAM_CODE = r"""import torch
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

    for param_group in training_manager_instance.optimizer_adam.param_groups:
        param_group['initial_lr'] = training_manager_instance.base_learning_rate_adam
        param_group['lr'] = training_manager_instance.base_learning_rate_adam
        
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(training_manager_instance.optimizer_adam, T_max=max(epochs, 1), eta_min=eta_min)
    stop_reason = f"Completed all {epochs} epochs."
    non_finite_streak = 0
    training_manager_instance.completed_adam_epochs = 0
    
    for epoch in range(epochs):
        training_manager_instance.global_step += 1
        is_resampling_step = False
        if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
            domains_data = resample_fn()
            is_resampling_step = True

        training_manager_instance.optimizer_adam.zero_grad(set_to_none=True)
        loss = training_manager_instance.compute_loss(
            domains_data,
            update_relobralo=True,
            is_resampling_step=is_resampling_step
        )
        
        if not torch.isfinite(loss):
            non_finite_streak += 1
            if non_finite_streak >= max_non_finite_streak:
                stop_reason = f"Early stop due to non-finite Loss (NaN/Inf) for {max_non_finite_streak} consecutive steps."
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
            if getattr(training_manager_instance, 'relobralo_balancer', None) is not None:
                lmb = bd.get('relobralo_lambdas', {})
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e} | "
                    f"Lambda(pde/a/bn/ht) = {lmb.get('loss_pde', 1.0):.2f}/{lmb.get('loss_a', 1.0):.2f}/{lmb.get('loss_bn', 1.0):.2f}/{lmb.get('loss_ht', 1.0):.2f}"
                )
            else:
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
                )
            
        if training_manager_instance.visualizer and (epoch + 1) % training_manager_instance.visualizer_update_interval == 0:
            training_manager_instance.visualizer.save_frame(training_manager_instance.global_step, current_loss_value, domains_data)
            
    print(f"[*] ADAM FINISHED. Reason: {stop_reason}")
    return domains_data
"""


TRAIN_LBFGS_CODE = r"""import torch
import torch.optim as optim


def train_lbfgs(training_manager_instance, max_iterations, domains_data):
    if max_iterations <= 0:
        return domains_data

    model = training_manager_instance.model
    model.train()
    pinn = training_manager_instance.electro_magnetic_pinn
    t_cfg = getattr(pinn, 'training_config', None)
    log_interval = getattr(t_cfg, 'lbfgs_log_interval', 25)

    training_manager_instance.optimizer_lbfgs = optim.LBFGS(
        model.networks.parameters(),
        lr=training_manager_instance.lbfgs_learning_rate,
        max_iter=max_iterations,
        max_eval=int(max_iterations * 1.25),
        tolerance_grad=training_manager_instance.lbfgs_tolerance_gradient,
        tolerance_change=training_manager_instance.lbfgs_tolerance_change,
        history_size=training_manager_instance.lbfgs_history_size,
        line_search_fn="strong_wolfe"
    )

    lbfgs_counter = [0]
    
    def closure():
        training_manager_instance.optimizer_lbfgs.zero_grad(set_to_none=True)
        loss = training_manager_instance.compute_loss(
            domains_data,
            update_relobralo=False,
            is_resampling_step=False
        )
        if not torch.isfinite(loss):
            return loss
        loss.backward()
        
        lbfgs_counter[0] += 1
        if lbfgs_counter[0] == 1 or lbfgs_counter[0] % log_interval == 0:
            bd = training_manager_instance.last_loss_breakdown
            print(
                f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {loss.item():.4e} | "
                f"PDE = {bd.get('pde_loss', 0.0):.3e}"
            )
        return loss
        
    training_manager_instance.optimizer_lbfgs.step(closure)
    return domains_data
"""


EXECUTE_TRAINING_PROCESS_CODE = r"""def execute_training_process(pinn_instance):
    if pinn_instance.pinn_architecture_instance is None:
        pinn_instance._build_system()

    def resample_data_fn():
        return pinn_instance.collocation_sampler_instance.generate_all_domain_points()

    domains_data = resample_data_fn()
    t_cfg = pinn_instance.training_config
    tm = pinn_instance.training_manager_instance

    pre_ep = getattr(t_cfg, 'pretrain_nu_epochs', 300)
    pre_lr = getattr(t_cfg, 'pretrain_nu_lr', 1e-2)
    tm.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)

    total_adam_epochs = t_cfg.number_of_adam_epochs
    warmup_ratio = float(getattr(t_cfg, 'linear_warmup_epochs_ratio', 0.3))
    epochs_warmup = int(total_adam_epochs * warmup_ratio) if getattr(t_cfg, 'enable_linear_warmup_stage', True) else 0
    epochs_adaptation = (total_adam_epochs - epochs_warmup) if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) else 0

    if getattr(t_cfg, 'enable_linear_warmup_stage', True) and epochs_warmup > 0:
        print(f"\n--- [STAGE 1A] Adam Linear Warm-up ({epochs_warmup} Epochs) ---")
        if getattr(tm, 'relobralo_balancer', None) is not None:
            tm.relobralo_balancer.reset_anchors(reset_weights=False)
        domains_data = tm.train_adam(
            epochs=epochs_warmup, domains_data=domains_data,
            dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn, resample_interval=pinn_instance.sampler_config.resample_interval
        )

        if getattr(t_cfg, 'enable_lbfgs_after_warmup', False):
            lbfgs_warmup_iters = int(getattr(t_cfg, 'lbfgs_warmup_iterations', getattr(t_cfg, 'number_of_lbfgs_iterations', 1500)))
            if lbfgs_warmup_iters > 0:
                print(f"\n--- [STAGE 1A-LBFGS] L-BFGS Warm-up Refinement ({lbfgs_warmup_iters} Iterations) ---")
                domains_data = tm.train_lbfgs(max_iterations=lbfgs_warmup_iters, domains_data=domains_data)

    if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) and epochs_adaptation > 0:
        fpi_iters = int(getattr(t_cfg, 'fpi_outer_iterations', 10))
        alpha_damp = float(getattr(t_cfg, 'fpi_alpha_damping', 1.0))
        fpi_tol = float(getattr(t_cfg, 'fpi_tolerance', 1e-5))
        epochs_per_fpi = max(1, epochs_adaptation // fpi_iters)

        fit_ep = getattr(t_cfg, 'fit_nu_epochs', 250)
        fit_lr = getattr(t_cfg, 'fit_nu_lr', 3e-3)

        print(f"\n--- [STAGE 1B] FPI Outer Loop ({fpi_iters} iterations, {epochs_per_fpi} epochs/iter) ---")
        for f_iter in range(fpi_iters):
            print(f"\n>>> [FPI OUTER LOOP {f_iter + 1}/{fpi_iters}] <<<")
            if getattr(tm, 'relobralo_balancer', None) is not None:
                tm.relobralo_balancer.reset_anchors(reset_weights=False)
            domains_data = tm.train_adam(
                epochs=epochs_per_fpi, domains_data=domains_data,
                dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn, resample_interval=pinn_instance.sampler_config.resample_interval
            )
            max_diff = tm.fit_nu_network_step(
                domains_data, alpha_damping=alpha_damp, epochs=fit_ep, lr=fit_lr
            )
            print(f"[*] Outer FPI Iter {f_iter + 1}: Max Nu Diff = {max_diff:.4e}")

            if getattr(t_cfg, 'enable_lbfgs_after_fpi_step', False):
                lbfgs_fpi_iters = int(getattr(t_cfg, 'lbfgs_fpi_step_iterations', getattr(t_cfg, 'number_of_lbfgs_iterations', 1500)))
                if lbfgs_fpi_iters > 0:
                    print(f"\n--- [STAGE 1B-LBFGS] L-BFGS FPI Step Refinement ({lbfgs_fpi_iters} Iterations) ---")
                    domains_data = tm.train_lbfgs(max_iterations=lbfgs_fpi_iters, domains_data=domains_data)

            if max_diff < fpi_tol:
                print(f"[OK] FPI Early convergence at iteration {f_iter + 1}!")
                break

    if getattr(t_cfg, 'enable_lbfgs_refinement_stage', True) and t_cfg.number_of_lbfgs_iterations > 0:
        if getattr(pinn_instance.sampler_config, 'resample_points_before_lbfgs_stage', True):
            domains_data = resample_data_fn()

        lbfgs_iters = int(t_cfg.number_of_lbfgs_iterations)
        print(f"\n--- [STAGE 2] L-BFGS Refinement ({lbfgs_iters} Iterations) ---")
        tm.train_lbfgs(max_iterations=lbfgs_iters, domains_data=domains_data)

    if pinn_instance.visualizer_instance:
        pinn_instance.visualizer_instance.generate_animations()
"""


SAVE_CHECKPOINT_CODE = r"""import pickle
from pathlib import Path


def get_project_root():
    current_file = Path(__file__).resolve()
    return current_file.parents[3]


def resolve_checkpoint_filepath(filepath=None):
    root_dir = get_project_root()
    default_dir = root_dir / "checkpoints"

    if filepath is None:
        return default_dir / "pinn_model_checkpoint.pkl"

    path_obj = Path(filepath)

    if path_obj.is_absolute():
        return path_obj

    if path_obj.parent != Path("."):
        return (root_dir / path_obj).resolve()

    return default_dir / path_obj.name


def save_checkpoint(pinn_instance, filepath=None):
    path = resolve_checkpoint_filepath(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arch = pinn_instance.pinn_architecture_instance
    tm = pinn_instance.training_manager_instance
    sim = pinn_instance.simulation_domain_engine_instance

    checkpoint_data = {
        "model_state_dict": arch.state_dict() if arch else None,
        "optimizer_adam_state_dict": (
            tm.optimizer_adam.state_dict() if tm and tm.optimizer_adam else None
        ),
        "optimizer_lbfgs_state_dict": (
            tm.optimizer_lbfgs.state_dict()
            if tm and tm.optimizer_lbfgs
            else None
        ),
        "relobralo_state_dict": (
            tm.relobralo_balancer.state_dict()
            if tm and getattr(tm, "relobralo_balancer", None)
            else None
        ),
        "completed_adam_epochs": getattr(tm, "completed_adam_epochs", 0)
        if tm
        else 0,
        "global_step": getattr(tm, "global_step", 0) if tm else 0,
        "sampler_config": pinn_instance.sampler_config,
        "training_config": pinn_instance.training_config,
        "visualization_config": pinn_instance.visualization_config,
        "physics_config": sim.physics_config if sim else None,
        "x_boundaries_tuple": sim.x_boundaries_tuple if sim else None,
        "y_boundaries_tuple": sim.y_boundaries_tuple if sim else None,
        "loss_history": (
            pinn_instance.visualizer_instance.loss_history
            if pinn_instance.visualizer_instance
            else []
        ),
        "epoch_history": (
            pinn_instance.visualizer_instance.epoch_history
            if pinn_instance.visualizer_instance
            else []
        ),
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"[*] Successfully saved single-file pickle checkpoint to: {path}")
"""


LOAD_CHECKPOINT_CODE = r"""import pickle
from pathlib import Path


def get_project_root():
    current_file = Path(__file__).resolve()
    return current_file.parents[3]


def resolve_checkpoint_filepath(filepath=None):
    root_dir = get_project_root()
    default_dir = root_dir / "checkpoints"

    if filepath is None:
        return default_dir / "pinn_model_checkpoint.pkl"

    path_obj = Path(filepath)

    if path_obj.is_absolute():
        return path_obj

    if path_obj.parent != Path("."):
        return (root_dir / path_obj).resolve()

    return default_dir / path_obj.name


def load_checkpoint(pinn_instance, filepath=None):
    path = resolve_checkpoint_filepath(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    with open(path, "rb") as f:
        checkpoint_data = pickle.load(f)

    if "sampler_config" in checkpoint_data:
        pinn_instance.sampler_config = checkpoint_data["sampler_config"]
    if "training_config" in checkpoint_data:
        pinn_instance.training_config = checkpoint_data["training_config"]
    if "visualization_config" in checkpoint_data:
        pinn_instance.visualization_config = checkpoint_data[
            "visualization_config"
        ]

    sim = pinn_instance.simulation_domain_engine_instance
    if sim:
        if "physics_config" in checkpoint_data:
            sim.physics_config = checkpoint_data["physics_config"]
        if "x_boundaries_tuple" in checkpoint_data:
            sim.x_boundaries_tuple = checkpoint_data["x_boundaries_tuple"]
        if "y_boundaries_tuple" in checkpoint_data:
            sim.y_boundaries_tuple = checkpoint_data["y_boundaries_tuple"]

    if pinn_instance.pinn_architecture_instance is None:
        pinn_instance._build_system()

    if (
        checkpoint_data.get("model_state_dict")
        and pinn_instance.pinn_architecture_instance
    ):
        pinn_instance.pinn_architecture_instance.load_state_dict(
            checkpoint_data["model_state_dict"]
        )

    tm = pinn_instance.training_manager_instance
    if tm:
        if checkpoint_data.get("optimizer_adam_state_dict") and tm.optimizer_adam:
            tm.optimizer_adam.load_state_dict(
                checkpoint_data["optimizer_adam_state_dict"]
            )
        if (
            checkpoint_data.get("optimizer_lbfgs_state_dict")
            and tm.optimizer_lbfgs
        ):
            tm.optimizer_lbfgs.load_state_dict(
                checkpoint_data["optimizer_lbfgs_state_dict"]
            )
        if (
            checkpoint_data.get("relobralo_state_dict")
            and getattr(tm, "relobralo_balancer", None)
        ):
            tm.relobralo_balancer.load_state_dict(
                checkpoint_data["relobralo_state_dict"]
            )
        tm.completed_adam_epochs = checkpoint_data.get(
            "completed_adam_epochs", 0
        )
        tm.global_step = checkpoint_data.get("global_step", 0)

    vis = pinn_instance.visualizer_instance
    if vis:
        vis.loss_history = checkpoint_data.get("loss_history", [])
        vis.epoch_history = checkpoint_data.get("epoch_history", [])

    print(
        f"[*] Successfully loaded single-file pickle checkpoint from: {path}"
    )
    return pinn_instance
"""


EXCUTE6_CODE = r"""import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn

try:
    import ansys.aedt.core
except ImportError:
    if "google.colab" in sys.modules:
        print("[*] Installing pyaedt library on Colab...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyaedt"])

re_simulation_fem = False
re_train_pinn = True

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, ".."))
pinn_module_path = os.path.join(parent_directory, "src", "physics_informed_model")

if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)
if pinn_module_path not in sys.path:
    sys.path.insert(0, pinn_module_path)

from src.ansys_aedt.maxwell_2d.toy_model_1.toy_model_1 import ToyModel1
from electro_magnetic_pinn import ElectroMagneticPINN
from src.ansys_aedt.maxwell_2d.toy_model_1.plot_flux_density import ANSYS_FEM_HEX_COLORS
from checkpoint_engine.save_checkpoint import resolve_checkpoint_filepath

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5
})


def main():
    output_dir = Path(parent_directory) / "figure_high_precision_simulation_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "toymodel1_relobralo"

    yoke_outline = [[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
    magnet_left_outline = [[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]]
    magnet_right_outline = [[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]]
    region_outline = [[-0.12, -0.12], [0.12, -0.12], [0.12, 0.12], [-0.12, 0.12]]

    magnet_left_coercive_dir = (0, 1)
    magnet_right_coercive_dir = (0, -1)

    magnet_material_name = "NdFe30"
    yoke_material_name = "steel_1008"

    print("[1/4] ANSYS MAXWELL FEM SIMULATION PROCESSING")
    toy_model = ToyModel1()
    fem_config = toy_model.configuration

    fem_config.name = "toy_model_1"
    fem_config.version = "2025.2"
    fem_config.solution_type = "Magnetostatic"

    fem_config.geometry.yoke.outline = yoke_outline
    fem_config.geometry.magnet_left.outline = magnet_left_outline
    fem_config.geometry.magnet_left.coercive_direction = magnet_left_coercive_dir
    fem_config.geometry.magnet_right.outline = magnet_right_outline
    fem_config.geometry.magnet_right.coercive_direction = magnet_right_coercive_dir
    fem_config.geometry.region.outline = region_outline

    fem_config.materials.yoke_material = yoke_material_name
    fem_config.materials.magnet_material_left = "NdFe30 - Copy"
    fem_config.materials.magnet_material_right = "NdFe30 - Copy (2)"
    fem_config.materials.region_material = "vacuum"

    fem_config.analysis.setup_name = "Setup1"
    fem_config.analysis.max_passes = 10
    fem_config.analysis.min_passes = 2
    fem_config.analysis.percent_error = 1.0
    fem_config.analysis.max_element_length = "0.025mm"
    
    fld_path = Path(parent_directory) / "src" / "ansys_aedt" / "maxwell_2d" / "toy_model_1" / "flux_density.fld"
    fem_config.analysis.b_fld_path = str(fld_path)
    fem_config.analysis.grid_min = ["-120mm", "-120mm", "0mm"]
    fem_config.analysis.grid_max = ["120mm", "120mm", "0mm"]
    fem_config.analysis.grid_step = ["1mm", "1mm", "0mm"]

    if re_simulation_fem:
        print("[*] re_simulation_fem = True: Creating model and running FEM simulation on Ansys Maxwell...")
        toy_model.create_model()
        toy_model.run_simulation()
        toy_model.export_flux_density(export_path=str(fld_path))
    else:
        print("[*] re_simulation_fem = False: Reading FEM field data directly from .fld file...")
        if not fld_path.exists():
            raise FileNotFoundError(f"FEM field data file not found at: {fld_path}. Please set re_simulation_fem = True to simulate.")
        toy_model.export_flux_density(export_path=str(fld_path))

    field_record = toy_model.record.flux_density_field
    fem_x_mm = field_record.x
    fem_y_mm = field_record.y
    fem_b_mag = field_record.b_mag

    print("[2/4] PINN MODEL SETUP (ELECTROMAGNETIC PINN WITH GLOBAL RELOBRALO)")
    pinn_model = ElectroMagneticPINN()
    sim_domain = pinn_model.simulation_domain_engine_instance

    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 960000.0
    sim_domain.physics_config.scale_nu0 = 795774.715459
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    pinn_model.sampler_config.dynamic_resampling = True
    pinn_model.sampler_config.resample_interval = 10
    pinn_model.sampler_config.resample_points_before_lbfgs_stage = True

    pinn_model.visualization_config.active = True
    pinn_model.visualization_config.display_colab_frames = True
    pinn_model.visualization_config.update_interval = 200

    pinn_model.training_config.enable_relobralo = True
    pinn_model.training_config.relobralo_alpha = 0.999
    pinn_model.training_config.relobralo_temperature = 0.1
    pinn_model.training_config.relobralo_rho = 0.999

    pinn_model.training_config.enable_linear_warmup_stage = True
    pinn_model.training_config.enable_nonlinear_adaptation_stage = True
    pinn_model.training_config.enable_lbfgs_refinement_stage = True

    pinn_model.training_config.enable_lbfgs_after_warmup = True
    pinn_model.training_config.lbfgs_warmup_iterations = 1000

    pinn_model.training_config.enable_lbfgs_after_fpi_step = True
    pinn_model.training_config.lbfgs_fpi_step_iterations = 1000

    epochs_warmup_target = 5000
    fpi_iterations = 6
    epochs_per_fpi = 5000

    pinn_model.training_config.fpi_outer_iterations = fpi_iterations
    pinn_model.training_config.fpi_alpha_damping = 0.8
    pinn_model.training_config.fpi_tolerance = 1e-6

    pinn_model.training_config.number_of_adam_epochs = epochs_warmup_target + (fpi_iterations * epochs_per_fpi)
    pinn_model.training_config.linear_warmup_epochs_ratio = epochs_warmup_target / pinn_model.training_config.number_of_adam_epochs

    pinn_model.training_config.adam_learning_rate = 1e-3
    pinn_model.training_config.target_loss_threshold = 1e-7
    pinn_model.training_config.adam_eta_min = 1e-6
    pinn_model.training_config.grad_clip_max_norm = 1.0
    pinn_model.training_config.max_non_finite_streak = 5
    pinn_model.training_config.adam_log_interval = 100

    pinn_model.training_config.pretrain_nu_epochs = 1000
    pinn_model.training_config.pretrain_nu_lr = 5e-3
    pinn_model.training_config.fit_nu_epochs = 500
    pinn_model.training_config.fit_nu_lr = 3e-3

    pinn_model.training_config.number_of_lbfgs_iterations = 4000
    pinn_model.training_config.lbfgs_learning_rate = 0.5

    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name=yoke_material_name,
        outline=yoke_outline
    )
    iron_yoke.physics_config.relative_permeability = 1000.0
    iron_yoke.physics_config.coercive = [0.0, 0.0]
    iron_yoke.physics_config.current = 0.0
    iron_yoke.physics_config.is_nonlinear = True

    iron_yoke.training_config.pde_fading_beta = 80.0
    iron_yoke.training_config.sampling_density_multiplier = 1.0
    iron_yoke.training_config.number_of_interior_points = 1800
    iron_yoke.training_config.number_of_surface_points = 350

    iron_yoke.network_config.hidden_layers = 5
    iron_yoke.network_config.hidden_neurons = 128
    iron_yoke.network_config.activation_function = nn.Mish()
    iron_yoke.network_config.fourier_features = True
    iron_yoke.network_config.fourier_sigma = 1.3
    iron_yoke.network_config.fourier_mapping_size = 128

    iron_yoke.nu_network_config.hidden_layers = 4
    iron_yoke.nu_network_config.hidden_neurons = 64
    iron_yoke.nu_network_config.activation_function = nn.Mish()
    iron_yoke.nu_network_config.fourier_features = True
    iron_yoke.nu_network_config.fourier_sigma = 1.0
    iron_yoke.nu_network_config.fourier_mapping_size = 32
    iron_yoke.build_domain()

    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left",
        material_name=magnet_material_name,
        outline=magnet_left_outline,
        coercive_direction=magnet_left_coercive_dir
    )
    magnet_left.physics_config.is_nonlinear = False
    magnet_left.training_config.number_of_interior_points = 120
    magnet_left.training_config.number_of_surface_points = 60
    magnet_left.network_config.hidden_layers = 4
    magnet_left.network_config.hidden_neurons = 64
    magnet_left.network_config.activation_function = nn.Mish()
    magnet_left.network_config.fourier_features = True
    magnet_left.network_config.fourier_sigma = 1.0
    magnet_left.network_config.fourier_mapping_size = 32
    magnet_left.build_domain()

    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right",
        material_name=magnet_material_name,
        outline=magnet_right_outline,
        coercive_direction=magnet_right_coercive_dir
    )
    magnet_right.physics_config.is_nonlinear = False
    magnet_right.training_config.number_of_interior_points = 120
    magnet_right.training_config.number_of_surface_points = 60
    magnet_right.network_config.hidden_layers = 4
    magnet_right.network_config.hidden_neurons = 64
    magnet_right.network_config.activation_function = nn.Mish()
    magnet_right.network_config.fourier_features = True
    magnet_right.network_config.fourier_sigma = 1.0
    magnet_right.network_config.fourier_mapping_size = 32
    magnet_right.build_domain()

    bg_training_cfg = SimpleNamespace(
        pde_fading_beta=80.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=1000,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=128,
        activation_function=nn.Mish(),
        fourier_features=True,
        fourier_sigma=1.0,
        fourier_mapping_size=120
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    pinn_model.update_electromagnetic_pinn()

    if re_train_pinn:
        print("[*] re_train_pinn = True: Training PINN model with global ReLoBRaLo...")
        pinn_model.execute_training_process()
        print(f"[*] Saving PINN checkpoint: '{checkpoint_name}'...")
        pinn_model.save_checkpoint(checkpoint_name)
        
        if "google.colab" in sys.modules:
            checkpoint_path = resolve_checkpoint_filepath(checkpoint_name)
            if checkpoint_path.exists():
                print(f"[*] Google Colab environment detected. Preparing download: {checkpoint_path}...")
                from google.colab import files
                files.download(str(checkpoint_path))
            else:
                print(f"[!] Checkpoint file not found for download: {checkpoint_path}")
    else:
        print(f"[*] re_train_pinn = False: Loading existing PINN checkpoint '{checkpoint_name}'...")
        pinn_model.load_checkpoint(checkpoint_name)

    print("\n==================================================")
    print("[3/4] COMPUTING PINN B-FIELD ON SYNCHRONIZED GRID")
    print("==================================================")
    res_x = len(fem_x_mm)
    res_y = len(fem_y_mm)
    x_m = fem_x_mm / 1000.0
    y_m = fem_y_mm / 1000.0
    X_m, Y_m = np.meshgrid(x_m, y_m)

    eval_pts = torch.tensor(
        np.column_stack((X_m.ravel(), Y_m.ravel())),
        dtype=torch.float32,
        device=pinn_model.computation_device
    ).requires_grad_(True)

    _, B_x_pinn, B_y_pinn = pinn_model.evaluate_fields(eval_pts, return_numpy=True)
    pinn_b_mag = np.sqrt(B_x_pinn**2 + B_y_pinn**2).reshape(res_y, res_x)

    delta_b_mag = np.abs(pinn_b_mag - fem_b_mag)
    fem_b_max = np.max(fem_b_mag)
    norm_error_percent = (delta_b_mag / fem_b_max) * 100.0

    print("\n==================================================")
    print("[4/4] PLOTTING AND COMPARING FEM VS PINN (3 PANELS)")
    print("==================================================")
    cmap_ansys = mcolors.LinearSegmentedColormap.from_list("ansys_fem_interp", ANSYS_FEM_HEX_COLORS, N=100)
    levels = np.linspace(0.0, 2.0, 101)
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap_ansys.N, clip=True)

    fig, axs = plt.subplots(1, 3, figsize=(21, 6), dpi=300)
    extent = [fem_x_mm.min(), fem_x_mm.max(), fem_y_mm.min(), fem_y_mm.max()]

    im0 = axs[0].imshow(
        fem_b_mag,
        extent=extent,
        origin="lower",
        cmap=cmap_ansys,
        norm=norm,
        aspect="equal"
    )
    axs[0].set_title(r"Ansys Maxwell 2D FEM $|B|$ (Tesla)", fontweight="bold")
    axs[0].set_xlabel("X (mm)")
    axs[0].set_ylabel("Y (mm)")
    cbar0 = fig.colorbar(im0, ax=axs[0], pad=0.02)
    cbar0.set_label(r"$|B|$ (T)")

    im1 = axs[1].imshow(
        pinn_b_mag,
        extent=extent,
        origin="lower",
        cmap=cmap_ansys,
        norm=norm,
        aspect="equal"
    )
    axs[1].set_title(r"ElectroMagneticPINN $|B|$ (Tesla)", fontweight="bold")
    axs[1].set_xlabel("X (mm)")
    axs[1].set_ylabel("Y (mm)")
    cbar1 = fig.colorbar(im1, ax=axs[1], pad=0.02)
    cbar1.set_label(r"$|B|$ (T)")

    im2 = axs[2].imshow(
        norm_error_percent,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=10.0,
        aspect="equal"
    )
    axs[2].set_title(r"Normalized Error (%) [$\Delta|B| / B_{\mathrm{max}}$]", fontweight="bold")
    axs[2].set_xlabel("X (mm)")
    axs[2].set_ylabel("Y (mm)")
    cbar2 = fig.colorbar(im2, ax=axs[2], pad=0.02)
    cbar2.set_label("Normalized Error (%)")

    plt.tight_layout()
    save_fig_path = output_dir / "fem_vs_pinn_bmap_comparison_relobralo.png"
    plt.savefig(save_fig_path, dpi=300)
    print(f"[*] Saved comparison plot to: {save_fig_path}")

    print("[*] Displaying figure...")
    plt.show()


if __name__ == "__main__":
    main()
"""


def main():
    root = resolve_project_root()
    pinn_root = root / "src" / "physics_informed_model"

    updates = [
        (pinn_root / "simulation_domain_engine" / "material_domain" / "material_domain.py", MATERIAL_DOMAIN_CODE),
        (pinn_root / "simulation_domain_engine" / "global_physical_properties_evaluation.py", GLOBAL_PHYSICAL_PROPERTIES_CODE),
        (pinn_root / "physics_domain" / "physical_equations" / "interface_loss.py", INTERFACE_LOSS_CODE),
        (pinn_root / "electro_magnetic_pinn.py", ELECTRO_MAGNETIC_PINN_CODE),
        (pinn_root / "training_engine" / "relobralo_balancer.py", RELOBRALO_BALANCER_CODE),
        (pinn_root / "training_engine" / "__init__.py", TRAINING_ENGINE_INIT_CODE),
        (pinn_root / "training_engine" / "training_manager.py", TRAINING_MANAGER_CODE),
        (pinn_root / "training_engine" / "compute_loss.py", COMPUTE_LOSS_CODE),
        (pinn_root / "training_engine" / "train_adam.py", TRAIN_ADAM_CODE),
        (pinn_root / "training_engine" / "train_lbfgs.py", TRAIN_LBFGS_CODE),
        (pinn_root / "electro_magnetic_pinn_engine" / "execute_training_process.py", EXECUTE_TRAINING_PROCESS_CODE),
        (pinn_root / "checkpoint_engine" / "save_checkpoint.py", SAVE_CHECKPOINT_CODE),
        (pinn_root / "checkpoint_engine" / "load_checkpoint.py", LOAD_CHECKPOINT_CODE),
        (root / "scenarios" / "excute6.py", EXCUTE6_CODE)
    ]

    for target_path, content in updates:
        write_file(target_path, content)

    print("[*] All ReLoBRaLo and MaterialDomain updates applied successfully.")


if __name__ == "__main__":
    main()
