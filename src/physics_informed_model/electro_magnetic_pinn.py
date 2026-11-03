import torch
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


class ElectroMagneticPINN:
    def __init__(self):
        self.sampler_config = SimpleNamespace(
            dynamic_resampling=True,
            resample_interval=100,
            resample_points_before_lbfgs_stage=True
        )
        
        self.training_config = SimpleNamespace(
            enable_dynamic_weighting=True,
            dynamic_weight_method="grad_norm", # "grad_norm" hoặc "uncertainty"
            dynamic_weight_update_interval=50,
            dynamic_weight_alpha=0.1,
            max_interface_weight_clamp=20.0,

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
        return _build_system(self)

    def evaluate_fields(self, points_tensor, return_numpy=False):
        return evaluate_fields(self, points_tensor, return_numpy=return_numpy)

    def plot_fields(self, resolution=180, save_path=None, show=False, figsize=(12, 10)):
        return plot_fields(self, resolution=resolution, save_path=save_path, show=show, figsize=figsize)

    def execute_training_process(self):
        return execute_training_process(self)
