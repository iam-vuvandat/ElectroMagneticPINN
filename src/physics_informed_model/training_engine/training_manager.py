import time
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

    def train_adam(
        self,
        epochs,
        domains_data,
        dynamic_resampling=False,
        resample_fn=None,
        resample_interval=100,
        lr_begin=None,
        lr_finish=None
    ):
        return train_adam(
            self,
            epochs,
            domains_data,
            dynamic_resampling=dynamic_resampling,
            resample_fn=resample_fn,
            resample_interval=resample_interval,
            lr_begin=lr_begin,
            lr_finish=lr_finish
        )

    def train_lbfgs(self, max_iterations, domains_data):
        return train_lbfgs(self, max_iterations, domains_data)