from pathlib import Path

current_path = Path(__file__).resolve().parent
project_root = current_path if (current_path / "src").exists() else Path.cwd()

files_to_write = {}

files_to_write["src/physics_informed_model/training_engine/curriculum_scheduler.py"] = '''import math


class CurriculumWeightScheduler:
    def __init__(
        self,
        total_warmup_epochs,
        stage_order=("loss_ht", "loss_a", "loss_bn", "loss_pde"),
        steepness=15.0,
        min_weights=None,
        max_weights=None,
        couple_bc_with_az=True
    ):
        self.total_warmup_epochs = max(int(total_warmup_epochs), 0)
        self.stage_order = tuple(stage_order)
        self.steepness = float(steepness)
        self.couple_bc_with_az = bool(couple_bc_with_az)

        default_min = {
            "loss_ht": 1.0,
            "loss_a": 0.02,
            "loss_bc": 0.05,
            "loss_bn": 0.01,
            "loss_pde": 1e-3
        }
        default_max = {k: 1.0 for k in default_min}

        if min_weights:
            default_min.update({k: float(v) for k, v in min_weights.items()})
        if max_weights:
            default_max.update({k: float(v) for k, v in max_weights.items()})

        self.min_weights = default_min
        self.max_weights = default_max

        num_stages = max(len(self.stage_order), 1)
        self.stage_centers = {
            key: float(idx) / float(num_stages)
            for idx, key in enumerate(self.stage_order)
        }

    def is_active(self, step):
        return self.total_warmup_epochs > 0 and int(step) < self.total_warmup_epochs

    def _smooth_gate(self, progress, center, min_w, max_w):
        if center <= 0.0:
            return float(max_w)
        exponent = max(-50.0, min(50.0, -self.steepness * (progress - center)))
        sigmoid = 1.0 / (1.0 + math.exp(exponent))
        return float(min_w + (max_w - min_w) * sigmoid)

    def get_weights(self, step, keys=None):
        if not self.is_active(step):
            return None

        progress = float(step) / float(self.total_warmup_epochs)
        weights = {}

        for key, center in self.stage_centers.items():
            min_w = self.min_weights.get(key, 1e-3)
            max_w = self.max_weights.get(key, 1.0)
            weights[key] = self._smooth_gate(progress, center, min_w, max_w)

        if self.couple_bc_with_az and "loss_bc" not in self.stage_centers:
            az_center = self.stage_centers.get("loss_a", 0.25)
            weights["loss_bc"] = self._smooth_gate(
                progress,
                az_center,
                self.min_weights.get("loss_bc", 0.05),
                self.max_weights.get("loss_bc", 1.0)
            )

        if keys is not None:
            return {k: float(weights.get(k, 1.0)) for k in keys}
        return weights

    def state_dict(self):
        return {
            "total_warmup_epochs": int(self.total_warmup_epochs),
            "stage_order": list(self.stage_order),
            "steepness": float(self.steepness),
            "min_weights": dict(self.min_weights),
            "max_weights": dict(self.max_weights),
            "couple_bc_with_az": bool(self.couple_bc_with_az),
            "stage_centers": dict(self.stage_centers)
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.total_warmup_epochs = int(state.get("total_warmup_epochs", self.total_warmup_epochs))
        if "stage_order" in state:
            self.stage_order = tuple(state["stage_order"])
        self.steepness = float(state.get("steepness", self.steepness))
        if "min_weights" in state:
            self.min_weights = {k: float(v) for k, v in state["min_weights"].items()}
        if "max_weights" in state:
            self.max_weights = {k: float(v) for k, v in state["max_weights"].items()}
        self.couple_bc_with_az = bool(state.get("couple_bc_with_az", self.couple_bc_with_az))
        if "stage_centers" in state:
            self.stage_centers = {k: float(v) for k, v in state["stage_centers"].items()}
'''

files_to_write["src/physics_informed_model/training_engine/__init__.py"] = '''from .training_manager import TrainingManager
from .resolve_domain_net_key import resolve_domain_net_key
from .pretrain_nu_networks import pretrain_nu_networks
from .fit_nu_network_step import fit_nu_network_step
from .compute_loss import compute_loss
from .train_adam import train_adam
from .train_lbfgs import train_lbfgs
from .relobralo_balancer import ReLoBRaLoBalancer
from .curriculum_scheduler import CurriculumWeightScheduler

__all__ = [
    "TrainingManager",
    "resolve_domain_net_key",
    "pretrain_nu_networks",
    "fit_nu_network_step",
    "compute_loss",
    "train_adam",
    "train_lbfgs",
    "ReLoBRaLoBalancer",
    "CurriculumWeightScheduler"
]
'''

files_to_write["src/physics_informed_model/electro_magnetic_pinn.py"] = '''import torch
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

            enable_curriculum_weighting=False,
            curriculum_steepness=15.0,
            curriculum_order=("loss_ht", "loss_a", "loss_bn", "loss_pde"),
            curriculum_min_weights=None,
            curriculum_max_weights=None,

            enable_relobralo=False,
            relobralo_alpha=0.999,
            relobralo_temperature=0.1,
            relobralo_rho=0.999,

            adam_learning_rate=1e-3,
            adam_lr_scheduler="linear",
            number_of_adam_epochs=2500,
            target_loss_threshold=1e-5,

            fpi_outer_iterations=10,
            fpi_initial_alpha_damping=1.0,
            fpi_alpha_damping=0.3,
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
            az_vmin=-0.02,             
            az_vmax=0.01,             
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
'''

files_to_write["src/physics_informed_model/training_engine/training_manager.py"] = '''import time
import torch
import torch.optim as optim

from training_engine.resolve_domain_net_key import resolve_domain_net_key
from training_engine.pretrain_nu_networks import pretrain_nu_networks
from training_engine.fit_nu_network_step import fit_nu_network_step
from training_engine.compute_loss import compute_loss
from training_engine.train_adam import train_adam
from training_engine.train_lbfgs import train_lbfgs
from training_engine.relobralo_balancer import ReLoBRaLoBalancer
from training_engine.curriculum_scheduler import CurriculumWeightScheduler


class TrainingManager:
    def __init__(self, electro_magnetic_pinn):
        self.electro_magnetic_pinn = electro_magnetic_pinn

        t_cfg = self.electro_magnetic_pinn.training_config

        self.base_learning_rate_adam = getattr(t_cfg, 'adam_learning_rate', getattr(t_cfg, 'learning_rate_adam', 1e-3))
        self.adam_lr_scheduler = getattr(t_cfg, 'adam_lr_scheduler', 'linear')
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

        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        boundary_obj = getattr(sim_domain, 'boundary', None)
        enforce_method = getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor') if boundary_obj else 'hard_factor'
        balancer_keys = ("loss_pde", "loss_a", "loss_bn", "loss_ht", "loss_bc") if enforce_method == 'soft_loss' else ("loss_pde", "loss_a", "loss_bn", "loss_ht")

        self.enable_curriculum_weighting = bool(getattr(t_cfg, 'enable_curriculum_weighting', False))
        self.curriculum_scheduler = None
        if self.enable_curriculum_weighting:
            total_adam = int(getattr(t_cfg, 'number_of_adam_epochs', 0))
            warmup_ratio = float(getattr(t_cfg, 'linear_warmup_epochs_ratio', 0.3))
            warmup_epochs = int(total_adam * warmup_ratio) if getattr(t_cfg, 'enable_linear_warmup_stage', True) else 0
            self.curriculum_scheduler = CurriculumWeightScheduler(
                total_warmup_epochs=warmup_epochs,
                stage_order=getattr(t_cfg, 'curriculum_order', ("loss_ht", "loss_a", "loss_bn", "loss_pde")),
                steepness=float(getattr(t_cfg, 'curriculum_steepness', 15.0)),
                min_weights=getattr(t_cfg, 'curriculum_min_weights', None),
                max_weights=getattr(t_cfg, 'curriculum_max_weights', None)
            )

        self.enable_relobralo = bool(getattr(t_cfg, 'enable_relobralo', False))
        self.relobralo_balancer = None
        if self.enable_relobralo:
            self.relobralo_balancer = ReLoBRaLoBalancer(
                keys=balancer_keys,
                alpha=float(getattr(t_cfg, 'relobralo_alpha', 0.999)),
                temperature=float(getattr(t_cfg, 'relobralo_temperature', 0.1)),
                rho=float(getattr(t_cfg, 'relobralo_rho', 0.999))
            )

        self.last_loss_breakdown = {}
        self.completed_adam_epochs = 0
        self.global_step = 0
        self.current_fpi_step = 0

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
'''

files_to_write["src/physics_informed_model/training_engine/compute_loss.py"] = '''import torch
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
    total_bc_tensor = zero_tensor

    az_if_sum = 0.0
    bn_if_sum = 0.0
    ht_if_sum = 0.0
    weighted_if_sum = 0.0

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    pinn = training_manager_instance.electro_magnetic_pinn
    sim_domain = pinn.simulation_domain_engine_instance
    bg_domain = sim_domain.background_domain

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

    boundary_obj = getattr(sim_domain, 'boundary', None)
    enforce_method = getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor') if boundary_obj else 'hard_factor'
    bc_pts = domains_data.get("boundary_points")

    if enforce_method == 'soft_loss' and bc_pts is not None and bc_pts.shape[0] > 0:
        bc_pts_dev = bc_pts.to(device)
        pot_val_star = float(getattr(boundary_obj.physics_config, 'potential_value', 0.0)) / max(sim_domain.A0, 1e-12)
        phys_props = sim_domain.evaluate_global_physical_properties(bc_pts_dev)
        mat_id = phys_props["material_classification"].view(-1)

        sq_err_sum = zero_tensor
        total_bc_count = 0
        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, model)
            mask = (mat_id == float(d_id))
            if mask.any():
                sub_pts = bc_pts_dev[mask]
                Az_bc_pred = model.forward_network(net_key, sub_pts)
                sq_err_sum = sq_err_sum + torch.sum((Az_bc_pred - pot_val_star) ** 2)
                total_bc_count += sub_pts.shape[0]

        if total_bc_count > 0:
            raw_bc_loss = sq_err_sum / float(total_bc_count)
            w_bc = float(getattr(boundary_obj.training_config, 'boundary_loss_weight', 1.0))
            total_bc_tensor = w_bc * raw_bc_loss

    raw_losses = {
        "loss_pde": total_pde_tensor,
        "loss_a": total_az_tensor,
        "loss_bn": total_bn_tensor,
        "loss_ht": total_ht_tensor
    }
    if enforce_method == 'soft_loss':
        raw_losses["loss_bc"] = total_bc_tensor

    scheduler = getattr(training_manager_instance, 'curriculum_scheduler', None)
    balancer = getattr(training_manager_instance, 'relobralo_balancer', None)
    step = getattr(training_manager_instance, 'global_step', 0)

    curr_lambdas = scheduler.get_weights(step=step, keys=raw_losses.keys()) if scheduler is not None else None

    if curr_lambdas is not None:
        lambdas = curr_lambdas
    elif balancer is not None:
        if update_relobralo:
            lambdas = balancer.update(raw_losses, is_resampling_step=is_resampling_step)
        else:
            lambdas = balancer.get_weights(raw_losses.keys())
    else:
        lambdas = {k: 1.0 for k in raw_losses.keys()}

    total_loss = (
        lambdas["loss_pde"] * total_pde_tensor
        + lambdas["loss_a"] * total_az_tensor
        + lambdas["loss_bn"] * total_bn_tensor
        + lambdas["loss_ht"] * total_ht_tensor
    )
    if enforce_method == 'soft_loss':
        total_loss = total_loss + lambdas.get("loss_bc", 1.0) * total_bc_tensor

    training_manager_instance.last_loss_breakdown = {
        "pde_loss": total_pde_tensor.item(),
        "interface_Az": az_if_sum,
        "interface_Bn": bn_if_sum,
        "interface_Ht": ht_if_sum,
        "boundary_loss": total_bc_tensor.item(),
        "weighted_interface": weighted_if_sum,
        "relobralo_lambdas": dict(lambdas),
        "total": total_loss.item()
    }

    return total_loss
'''

files_to_write["src/physics_informed_model/checkpoint_engine/save_checkpoint.py"] = '''import pickle
from pathlib import Path
from types import SimpleNamespace
import torch


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


def _move_tensors_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_cpu(v) for v in obj)
    if isinstance(obj, SimpleNamespace):
        return SimpleNamespace(**{k: _move_tensors_to_cpu(v) for k, v in vars(obj).items()})
    return obj


def save_checkpoint(pinn_instance, filepath=None):
    path = resolve_checkpoint_filepath(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arch = pinn_instance.pinn_architecture_instance
    tm = pinn_instance.training_manager_instance
    sim = pinn_instance.simulation_domain_engine_instance
    vis = pinn_instance.visualizer_instance
    boundary_obj = getattr(sim, "boundary", None) if sim else None

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
        "curriculum_state_dict": (
            tm.curriculum_scheduler.state_dict()
            if tm and getattr(tm, "curriculum_scheduler", None)
            else None
        ),
        "relobralo_state_dict": (
            tm.relobralo_balancer.state_dict()
            if tm and getattr(tm, "relobralo_balancer", None)
            else None
        ),
        "completed_adam_epochs": getattr(tm, "completed_adam_epochs", 0) if tm else 0,
        "global_step": getattr(tm, "global_step", 0) if tm else 0,
        "current_fpi_step": getattr(tm, "current_fpi_step", 0) if tm else 0,
        "sampler_config": pinn_instance.sampler_config,
        "training_config": pinn_instance.training_config,
        "visualization_config": pinn_instance.visualization_config,
        "physics_config": sim.physics_config if sim else None,
        "x_boundaries_tuple": sim.x_boundaries_tuple if sim else None,
        "y_boundaries_tuple": sim.y_boundaries_tuple if sim else None,
        "boundary_geometry_config": boundary_obj.geometry_config if boundary_obj else None,
        "boundary_physics_config": boundary_obj.physics_config if boundary_obj else None,
        "boundary_training_config": boundary_obj.training_config if boundary_obj else None,
        "loss_history": vis.loss_history if vis else [],
        "epoch_history": vis.epoch_history if vis else [],
        "pde_loss_history": getattr(vis, "pde_loss_history", []) if vis else [],
        "az_loss_history": getattr(vis, "az_loss_history", []) if vis else [],
        "bn_loss_history": getattr(vis, "bn_loss_history", []) if vis else [],
        "ht_loss_history": getattr(vis, "ht_loss_history", []) if vis else [],
        "bc_loss_history": getattr(vis, "bc_loss_history", []) if vis else [],
        "w_pde_history": getattr(vis, "w_pde_history", []) if vis else [],
        "w_az_history": getattr(vis, "w_az_history", []) if vis else [],
        "w_bn_history": getattr(vis, "w_bn_history", []) if vis else [],
        "w_ht_history": getattr(vis, "w_ht_history", []) if vis else [],
        "w_bc_history": getattr(vis, "w_bc_history", []) if vis else [],
    }

    checkpoint_data = _move_tensors_to_cpu(checkpoint_data)

    with open(path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"[*] Successfully saved single-file pickle checkpoint to: {path}")
'''

files_to_write["src/physics_informed_model/checkpoint_engine/load_checkpoint.py"] = '''import io
import pickle
from pathlib import Path
import torch


class _DeviceUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location="cpu"):
        super().__init__(file)
        self.map_location = map_location

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        return super().find_class(module, name)


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
        pkl_candidate = Path(f"{path}.pkl")
        if pkl_candidate.exists():
            path = pkl_candidate
        elif path.suffix == ".pkl" and path.with_suffix("").exists():
            path = path.with_suffix("")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(path, "rb") as f:
        checkpoint_data = _DeviceUnpickler(f, map_location=target_device).load()

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
        boundary_obj = getattr(sim, "boundary", None)
        if boundary_obj:
            if checkpoint_data.get("boundary_geometry_config") is not None:
                boundary_obj.geometry_config = checkpoint_data["boundary_geometry_config"]
                boundary_obj.build_boundary()
            if checkpoint_data.get("boundary_physics_config") is not None:
                boundary_obj.physics_config = checkpoint_data["boundary_physics_config"]
            if checkpoint_data.get("boundary_training_config") is not None:
                boundary_obj.training_config = checkpoint_data["boundary_training_config"]

    if pinn_instance.pinn_architecture_instance is None:
        pinn_instance._build_system()
    else:
        pinn_instance.computation_device = target_device
        pinn_instance.pinn_architecture_instance.to(target_device)

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
            checkpoint_data.get("curriculum_state_dict")
            and getattr(tm, "curriculum_scheduler", None)
        ):
            tm.curriculum_scheduler.load_state_dict(
                checkpoint_data["curriculum_state_dict"]
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
        tm.current_fpi_step = checkpoint_data.get("current_fpi_step", 0)

    vis = pinn_instance.visualizer_instance
    if vis:
        vis.loss_history = checkpoint_data.get("loss_history", [])
        vis.epoch_history = checkpoint_data.get("epoch_history", [])
        vis.pde_loss_history = checkpoint_data.get("pde_loss_history", [])
        vis.az_loss_history = checkpoint_data.get("az_loss_history", [])
        vis.bn_loss_history = checkpoint_data.get("bn_loss_history", [])
        vis.ht_loss_history = checkpoint_data.get("ht_loss_history", [])
        vis.bc_loss_history = checkpoint_data.get("bc_loss_history", [])
        vis.w_pde_history = checkpoint_data.get("w_pde_history", [])
        vis.w_az_history = checkpoint_data.get("w_az_history", [])
        vis.w_bn_history = checkpoint_data.get("w_bn_history", [])
        vis.w_ht_history = checkpoint_data.get("w_ht_history", [])
        vis.w_bc_history = checkpoint_data.get("w_bc_history", [])

    print(
        f"[*] Successfully loaded single-file pickle checkpoint from: {path}"
    )
    return pinn_instance
'''

files_to_write["src/physics_informed_model/training_visualizer/training_visualizer.py"] = '''import os
import torch
import numpy as np

from training_visualizer.resolve_domain_net_key import resolve_domain_net_key
from training_visualizer.save_frame import save_frame
from training_visualizer.generate_gif import generate_gif
from training_visualizer.generate_mp4 import generate_mp4
from training_visualizer.generate_animations import generate_animations


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
        self.pde_loss_history = []
        self.az_loss_history = []
        self.bn_loss_history = []
        self.ht_loss_history = []
        self.bc_loss_history = []
        self.w_pde_history = []
        self.w_az_history = []
        self.w_bn_history = []
        self.w_ht_history = []
        self.w_bc_history = []

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        x_coords = np.linspace(self.x_bounds[0], self.x_bounds[1], self.resolution)
        y_coords = np.linspace(self.y_bounds[0], self.y_bounds[1], self.resolution)
        self.X_grid, self.Y_grid = np.meshgrid(x_coords, y_coords)
        self.eval_points_tensor = torch.tensor(
            np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel())),
            dtype=torch.float32
        )

    def record_history(self, epoch, loss_value, loss_breakdown=None):
        if loss_breakdown is None:
            tm = getattr(self.electro_magnetic_pinn, 'training_manager_instance', None)
            loss_breakdown = getattr(tm, 'last_loss_breakdown', {}) if tm is not None else {}
        if loss_breakdown is None:
            loss_breakdown = {}

        self.loss_history.append(float(loss_value))
        self.epoch_history.append(int(epoch))

        pde_val = max(float(loss_breakdown.get('pde_loss', loss_value)), 1e-16)
        az_val = max(float(loss_breakdown.get('interface_Az', 1e-16)), 1e-16)
        bn_val = max(float(loss_breakdown.get('interface_Bn', 1e-16)), 1e-16)
        ht_val = max(float(loss_breakdown.get('interface_Ht', 1e-16)), 1e-16)
        bc_val = max(float(loss_breakdown.get('boundary_loss', 1e-16)), 1e-16)

        self.pde_loss_history.append(pde_val)
        self.az_loss_history.append(az_val)
        self.bn_loss_history.append(bn_val)
        self.ht_loss_history.append(ht_val)
        self.bc_loss_history.append(bc_val)

        lmb = loss_breakdown.get('relobralo_lambdas', {})
        self.w_pde_history.append(float(lmb.get('loss_pde', 1.0)))
        self.w_az_history.append(float(lmb.get('loss_a', 1.0)))
        self.w_bn_history.append(float(lmb.get('loss_bn', 1.0)))
        self.w_ht_history.append(float(lmb.get('loss_ht', 1.0)))
        self.w_bc_history.append(float(lmb.get('loss_bc', 1.0)))

    def save_frame(self, epoch, loss_value, domains_data, loss_breakdown=None, fpi_step=None):
        return save_frame(
            self,
            epoch,
            loss_value,
            domains_data,
            loss_breakdown=loss_breakdown,
            fpi_step=fpi_step
        )

    def generate_gif(self):
        return generate_gif(self)

    def generate_mp4(self):
        return generate_mp4(self)

    def generate_animations(self):
        return generate_animations(self)
'''

files_to_write["src/physics_informed_model/training_visualizer/save_frame.py"] = '''import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY
from training_visualizer.resolve_domain_net_key import resolve_domain_net_key

ANSYS_FEM_HEX_COLORS = [
    '#0000FF', '#0036FF', '#006BFF', '#00A1FF', '#00D7FF',
    '#00FFF2', '#00FFBC', '#00FF86', '#00FF51', '#00FF1B',
    '#1BFF00', '#51FF00', '#86FF00', '#BCFF00', '#F2FF00',
    '#FFD700', '#FFA100', '#FF6B00', '#FF3600', '#FF0000'
]


def save_frame(visualizer_instance, epoch, loss_value, domains_data, loss_breakdown=None, fpi_step=None):
    tm = getattr(visualizer_instance.electro_magnetic_pinn, 'training_manager_instance', None)
    if loss_breakdown is None and tm is not None:
        loss_breakdown = getattr(tm, 'last_loss_breakdown', {})
    if loss_breakdown is None:
        loss_breakdown = {}
    if fpi_step is None:
        fpi_step = getattr(tm, 'current_fpi_step', 0) if tm is not None else 0

    if not visualizer_instance.epoch_history or visualizer_instance.epoch_history[-1] != int(epoch):
        visualizer_instance.record_history(epoch, loss_value, loss_breakdown=loss_breakdown)

    lmb = loss_breakdown.get('relobralo_lambdas', {})

    model = visualizer_instance.electro_magnetic_pinn.pinn_architecture_instance
    pde_evaluator = visualizer_instance.electro_magnetic_pinn.maxwell_pde_loss_instance
    interface_evaluator = visualizer_instance.electro_magnetic_pinn.interface_loss_instance
    sim_domain = visualizer_instance.electro_magnetic_pinn.simulation_domain_engine_instance
    vis_cfg = visualizer_instance.electro_magnetic_pinn.visualization_config

    model.eval()
    computation_device = next(model.parameters()).device
    eval_points = visualizer_instance.eval_points_tensor.to(computation_device).detach().clone().requires_grad_(True)

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

    A_z_numpy = A_z_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_x_numpy = B_x_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_y_numpy = B_y_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
    mu_r_numpy = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys_all, min=1e-6)).cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    residual_numpy = residual_total.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    bg_domain = sim_domain.background_domain

    if_coords_list = []
    err_az_list = []
    err_bn_list = []
    err_ht_list = []

    if interface_evaluator is not None:
        for key, d in domains_dict.items():
            domain = d["domain_object"]
            net_key = resolve_domain_net_key(domain, model)
            outer_pts = d.get("outer_surface_points", d.get("surface_points"))
            if outer_pts is not None and outer_pts.shape[0] > 0 and net_key != 'background':
                pts_req = outer_pts.to(computation_device).detach().clone().requires_grad_(True)
                e_az, e_bn, e_ht = interface_evaluator.compute_pointwise_pair_components(
                    key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=pts_req
                )
                if_coords_list.append(pts_req.detach().cpu().numpy())
                err_az_list.append(e_az.detach().cpu().numpy().ravel())
                err_bn_list.append(e_bn.detach().cpu().numpy().ravel())
                err_ht_list.append(e_ht.detach().cpu().numpy().ravel())

        for pair_key, pair_data in shared_dict.items():
            s_pts = pair_data.get("interface_points")
            if s_pts is not None and s_pts.shape[0] > 0:
                pts_req = s_pts.to(computation_device).detach().clone().requires_grad_(True)
                e_az, e_bn, e_ht = interface_evaluator.compute_pointwise_pair_components(
                    key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                    key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts_req
                )
                if_coords_list.append(pts_req.detach().cpu().numpy())
                err_az_list.append(e_az.detach().cpu().numpy().ravel())
                err_bn_list.append(e_bn.detach().cpu().numpy().ravel())
                err_ht_list.append(e_ht.detach().cpu().numpy().ravel())

    if if_coords_list:
        if_coords_all = np.vstack(if_coords_list)
        err_az_all = np.concatenate(err_az_list)
        err_bn_all = np.concatenate(err_bn_list)
        err_ht_all = np.concatenate(err_ht_list)
    else:
        if_coords_all = np.empty((0, 2))
        err_az_all = np.empty((0,))
        err_bn_all = np.empty((0,))
        err_ht_all = np.empty((0,))

    bc_pts = domains_data.get("boundary_points")
    bc_coords_np = np.empty((0, 2))
    err_bc_all = np.empty((0,))
    boundary_obj = getattr(sim_domain, 'boundary', None)
    enforce_method = getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor') if boundary_obj else 'hard_factor'

    if bc_pts is not None and bc_pts.shape[0] > 0:
        bc_pts_dev = bc_pts.to(computation_device)
        bc_coords_np = bc_pts.cpu().detach().numpy()
        pot_val_star = float(getattr(boundary_obj.physics_config, 'potential_value', 0.0)) / max(sim_domain.A0, 1e-12)
        bc_phys_props = sim_domain.evaluate_global_physical_properties(bc_pts_dev)
        bc_mat_id = bc_phys_props["material_classification"].view(-1)

        err_bc_list = []
        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, model)
            mask = (bc_mat_id == float(d_id))
            if mask.any():
                sub_pts = bc_pts_dev[mask]
                with torch.no_grad():
                    Az_bc_pred = model.forward_network(net_key, sub_pts)
                    sq_err = (Az_bc_pred - pot_val_star) ** 2
                err_bc_list.append(sq_err.cpu().view(-1).numpy())
        if err_bc_list:
            err_bc_all = np.concatenate(err_bc_list)

    fig, axs = plt.subplots(3, 4, figsize=(24, 15))
    n_levels = getattr(vis_cfg, 'contour_levels', 60)

    def draw_domain_outlines(ax_target, color='#111111', linestyle='-', linewidth=1.0, alpha=0.85):
        for dom in sim_domain.domains_list:
            if getattr(dom, 'domain_name', '') != 'background' and dom.geometry_config.outline is not None:
                outline = np.array(dom.geometry_config.outline)
                closed_poly = np.vstack([outline, outline[0]])
                ax_target.plot(
                    closed_poly[:, 0], closed_poly[:, 1],
                    color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=5
                )

    ax_pts = axs[0, 0]
    ax_pts.set_title("Collocation point", fontsize=11, fontweight='bold')
    domain_colors = ['#1a365d', '#742a2a', '#1c4532', '#4a5568', '#44337a', '#7b341e']
    pair_colors = ['#800000', '#004080', '#004d40', '#4a148c', '#bf360c']

    for idx, (key, d) in enumerate(domains_dict.items()):
        c_dom = domain_colors[idx % len(domain_colors)]
        int_pts = d.get("interior_points")
        if int_pts is not None and int_pts.shape[0] > 0:
            pts_np = int_pts.detach().cpu().numpy()
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=0.6, color=c_dom, marker='o', linewidths=0, alpha=0.55)
        surf_pts = d.get("outer_surface_points", d.get("surface_points"))
        if surf_pts is not None and surf_pts.shape[0] > 0:
            pts_np = surf_pts.detach().cpu().numpy()
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=1.2, color=c_dom, marker='o', linewidths=0, alpha=0.85)

    for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
        if_pts = pair_data.get("interface_points")
        if if_pts is not None and if_pts.shape[0] > 0:
            pts_np = if_pts.detach().cpu().numpy()
            c_pair = pair_colors[p_idx % len(pair_colors)]
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=1.2, color=c_pair, marker='o', linewidths=0, alpha=0.95)

    if bc_pts is not None and bc_pts.shape[0] > 0:
        ax_pts.scatter(bc_coords_np[:, 0], bc_coords_np[:, 1], s=1.5, color='#000000', marker='o', linewidths=0, alpha=0.9)

    draw_domain_outlines(ax_pts, color='#555555', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_pts.set_xlim(visualizer_instance.x_bounds)
    ax_pts.set_ylim(visualizer_instance.y_bounds)
    ax_pts.set_aspect('equal')

    disc_levels = getattr(vis_cfg, 'res_discrete_levels', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    res_min = vis_cfg.res_vmin if vis_cfg.res_vmin is not None else float(disc_levels[0])
    res_max = vis_cfg.res_vmax if vis_cfg.res_vmax is not None else float(disc_levels[-1])
    res_scale_mode = getattr(vis_cfg, 'res_scale', 'log')

    if res_scale_mode == 'log' and disc_levels is not None:
        base_cmap = plt.get_cmap(vis_cfg.res_cmap, len(disc_levels) - 1)
        norm_res = mcolors.BoundaryNorm(disc_levels, ncolors=base_cmap.N, clip=True)
        levels_res = disc_levels
    elif res_scale_mode == 'log':
        base_cmap = plt.get_cmap(vis_cfg.res_cmap)
        norm_res = LogNorm(vmin=res_min, vmax=res_max, clip=True)
        levels_res = np.logspace(np.log10(res_min), np.log10(res_max), n_levels)
    else:
        base_cmap = plt.get_cmap(vis_cfg.res_cmap)
        levels_res = np.linspace(res_min, res_max, n_levels)
        norm_res = Normalize(vmin=res_min, vmax=res_max, clip=True)

    res_abs = np.clip(np.abs(residual_numpy), res_min, res_max)
    contour_res = axs[0, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, res_abs,
        levels=levels_res, cmap=base_cmap, norm=norm_res
    )
    cbar_res = fig.colorbar(
        contour_res,
        ax=axs[0, 1],
        ticks=disc_levels if (res_scale_mode == 'log' and disc_levels is not None) else None
    )
    if res_scale_mode == 'log':
        cbar_res.ax.set_yscale('log')
    cbar_res.set_label("Residual Magnitude", fontsize=9)
    draw_domain_outlines(axs[0, 1], color='#FFFFFF', linestyle='--', linewidth=0.7, alpha=0.5)
    axs[0, 1].set_title("PDE residual map", fontsize=11, fontweight='bold')
    axs[0, 1].set_aspect('equal')

    has_bc_plot = (enforce_method == 'soft_loss') or ('loss_bc' in lmb)
    ax_loss = axs[0, 2]
    steps = visualizer_instance.epoch_history
    ax_loss.plot(steps, visualizer_instance.pde_loss_history, label="PDE loss", linewidth=1.2)
    ax_loss.plot(steps, visualizer_instance.az_loss_history, label=r"$A_z$ loss", linewidth=1.2)
    ax_loss.plot(steps, visualizer_instance.bn_loss_history, label=r"$B_n$ loss", linewidth=1.2)
    ax_loss.plot(steps, visualizer_instance.ht_loss_history, label=r"$H_t$ loss", linewidth=1.2)
    if has_bc_plot:
        ax_loss.plot(steps, visualizer_instance.bc_loss_history, label="BC loss", linewidth=1.2)
    ax_loss.set_yscale('log')
    ax_loss.set_title("Loss optimization", fontsize=11, fontweight='bold')
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, which="both", ls="--", alpha=0.25)
    ax_loss.legend(loc="upper right", fontsize=8.5)

    ax_w = axs[0, 3]
    if len(getattr(visualizer_instance, 'w_pde_history', [])) == len(steps):
        ax_w.plot(steps, visualizer_instance.w_pde_history, label="PDE", linewidth=1.2)
    ax_w.plot(steps, visualizer_instance.w_az_history, label=r"$A_z$", linewidth=1.2)
    ax_w.plot(steps, visualizer_instance.w_bn_history, label=r"$B_n$", linewidth=1.2)
    ax_w.plot(steps, visualizer_instance.w_ht_history, label=r"$H_t$", linewidth=1.2)
    if has_bc_plot:
        ax_w.plot(steps, visualizer_instance.w_bc_history, label="BC", linewidth=1.2)
    ax_w.set_title("Loss weight", fontsize=11, fontweight='bold')
    ax_w.set_xlabel("Step")
    ax_w.set_ylabel("Weight")
    ax_w.grid(True, which="both", ls="--", alpha=0.25)
    ax_w.legend(loc="upper right", fontsize=8.5)

    if_levels = getattr(vis_cfg, 'if_discrete_levels', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    if_min = getattr(vis_cfg, 'if_vmin', None)
    if_min = float(if_min) if if_min is not None else float(if_levels[0])
    if_max = getattr(vis_cfg, 'if_vmax', None)
    if_max = float(if_max) if if_max is not None else float(if_levels[-1])

    if res_scale_mode == 'log' and if_levels is not None:
        if_cmap = plt.get_cmap(vis_cfg.res_cmap, len(if_levels) - 1)
        norm_if = mcolors.BoundaryNorm(if_levels, ncolors=if_cmap.N, clip=True)
    elif res_scale_mode == 'log':
        if_cmap = plt.get_cmap(vis_cfg.res_cmap)
        norm_if = LogNorm(vmin=if_min, vmax=if_max, clip=True)
    else:
        if_cmap = plt.get_cmap(vis_cfg.res_cmap)
        norm_if = Normalize(vmin=if_min, vmax=if_max, clip=True)

    def plot_scatter_loss_map(ax_target, coords, err_vals, title_str, cbar_label):
        draw_domain_outlines(ax_target, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)
        if coords.shape[0] > 0 and err_vals.shape[0] > 0:
            err_clamped = np.clip(err_vals, if_min, if_max)
            sc = ax_target.scatter(
                coords[:, 0], coords[:, 1],
                c=err_clamped, cmap=if_cmap, norm=norm_if,
                s=6.0, marker='o', linewidths=0
            )
            cbar_if = fig.colorbar(
                sc,
                ax=ax_target,
                ticks=if_levels if (res_scale_mode == 'log' and if_levels is not None) else None
            )
            if res_scale_mode == 'log':
                cbar_if.ax.set_yscale('log')
            cbar_if.set_label(cbar_label, fontsize=9)
        ax_target.set_xlim(visualizer_instance.x_bounds)
        ax_target.set_ylim(visualizer_instance.y_bounds)
        ax_target.set_title(title_str, fontsize=11, fontweight='bold')
        ax_target.set_aspect('equal')

    plot_scatter_loss_map(axs[1, 0], bc_coords_np, err_bc_all, r"Boundary loss map", r"$(A_{z}^* - A_{z,\mathrm{bc}}^*)^2$")
    plot_scatter_loss_map(axs[1, 1], if_coords_all, err_az_all, r"$A_z$ interface loss map", r"$(A_{z,a}^* - A_{z,b}^*)^2$")
    plot_scatter_loss_map(axs[1, 2], if_coords_all, err_bn_all, r"$B_n$ interface loss map", r"$w_c (B_{n,a}^* - B_{n,b}^*)^2$")
    plot_scatter_loss_map(axs[1, 3], if_coords_all, err_ht_all, r"$H_t$ interface loss map", r"$w_c (\Delta H_t^* / \nu)^2$")

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

    contour_mur = axs[2, 0].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, mu_r_clamped, levels=levels_mur,
        cmap=vis_cfg.mur_cmap, norm=norm_mur
    )
    fig.colorbar(contour_mur, ax=axs[2, 0], label=lbl_mur)
    draw_domain_outlines(axs[2, 0], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 0].set_title("Permeability map", fontsize=11, fontweight='bold')
    axs[2, 0].set_aspect('equal')

    az_min = float(vis_cfg.az_vmin) if getattr(vis_cfg, 'az_vmin', None) is not None else float(np.min(A_z_numpy))
    az_max = float(vis_cfg.az_vmax) if getattr(vis_cfg, 'az_vmax', None) is not None else float(np.max(A_z_numpy))
    if az_min > az_max:
        az_min, az_max = az_max, az_min
    A_z_clamped = np.clip(A_z_numpy, az_min, az_max)

    if getattr(vis_cfg, 'az_scale', 'linear') == 'log' and az_min > 0.0 and az_max > az_min:
        levels_az = np.logspace(np.log10(az_min), np.log10(az_max), n_levels)
        norm_az = LogNorm(vmin=az_min, vmax=az_max)
    else:
        levels_az = np.linspace(az_min, az_max, n_levels) if az_min != az_max else n_levels
        norm_az = Normalize(vmin=az_min, vmax=az_max)

    contour_az = axs[2, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, A_z_clamped, levels=levels_az,
        cmap=vis_cfg.az_cmap, norm=norm_az
    )
    fig.colorbar(contour_az, ax=axs[2, 1], label=r"$A_z$ (Wb/m)")
    draw_domain_outlines(axs[2, 1], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 1].set_title(r"Az map", fontsize=11, fontweight='bold')
    axs[2, 1].set_aspect('equal')

    b_unit = getattr(vis_cfg, 'b_mag_unit', 'Tesla')
    b_scale = 1000.0 if b_unit == "mTesla" else 1.0
    B_mag_plot = B_mag_numpy * b_scale
    unit_str = "mTesla" if b_unit == "mTesla" else "T"

    b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
    b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else (1800.0 if b_unit == "mTesla" else 1.8)
    b_mag_clamped = np.clip(B_mag_plot, b_min, b_max)
    num_div = getattr(vis_cfg, 'b_mag_number_of_division', None)

    if getattr(vis_cfg, 'b_mag_cmap', 'jet') == "ansys_fem":
        if num_div is not None and num_div > 0:
            cmap_b = mcolors.LinearSegmentedColormap.from_list("ansys_fem_interp", ANSYS_FEM_HEX_COLORS, N=num_div)
            levels_b = np.linspace(b_min, b_max, num_div + 1)
            norm_b = mcolors.BoundaryNorm(levels_b, ncolors=cmap_b.N, clip=True)
        else:
            cmap_b = mcolors.ListedColormap(ANSYS_FEM_HEX_COLORS)
            levels_b = np.linspace(b_min, b_max, 21)
            norm_b = mcolors.BoundaryNorm(levels_b, ncolors=cmap_b.N, clip=True)
    else:
        cmap_b = vis_cfg.b_mag_cmap
        levels_b = np.linspace(b_min, b_max, n_levels)
        norm_b = Normalize(vmin=b_min, vmax=b_max)

    contour_b = axs[2, 2].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=cmap_b, norm=norm_b
    )
    fig.colorbar(contour_b, ax=axs[2, 2], label=f"|B| ({unit_str})")
    draw_domain_outlines(axs[2, 2], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 2].set_title(r"|B| map", fontsize=11, fontweight='bold')
    axs[2, 2].set_aspect('equal')

    contour_b_bg = axs[2, 3].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=cmap_b, norm=norm_b, alpha=0.3
    )
    fig.colorbar(contour_b_bg, ax=axs[2, 3], label=f"|B| ({unit_str})")
    step = max(1, visualizer_instance.resolution // 20)
    B_sub = np.maximum(B_mag_numpy[::step, ::step], 1e-12)
    axs[2, 3].quiver(
        visualizer_instance.X_grid[::step, ::step], visualizer_instance.Y_grid[::step, ::step],
        B_x_numpy[::step, ::step] / B_sub, B_y_numpy[::step, ::step] / B_sub,
        scale=30.0, color='black', pivot='mid'
    )
    draw_domain_outlines(axs[2, 3], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 3].set_title("B vector map", fontsize=11, fontweight='bold')
    axs[2, 3].set_aspect('equal')

    fig.suptitle(f"Training process: Step {epoch}; FPI step {fpi_step}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    frame_name = os.path.join(visualizer_instance.output_directory, f"frame_{len(visualizer_instance.frame_paths) + 1:05d}.png")
    plt.savefig(frame_name, dpi=100)
    plt.close(fig)

    visualizer_instance.frame_paths.append(frame_name)
    model.train()

    if getattr(vis_cfg, 'display_colab_frames', False):
        try:
            from IPython.display import Image, display, clear_output
            clear_output(wait=True)
            print(f"[*] Live Colab Frame at Step {epoch} (FPI step {fpi_step}): {frame_name}")
            display(Image(filename=frame_name))
        except Exception:
            pass
'''

for rel_path, content in files_to_write.items():
    target_path = project_root / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    print(f"[+] Updated: {target_path}")

print("[*] Successfully applied CurriculumWeightScheduler update across all modules.")