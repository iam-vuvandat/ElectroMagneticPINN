import time
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

        last_param = None
        for p in model.networks.parameters():
            if p.requires_grad:
                last_param = p

        if last_param is None:
            return

        grad_pde = torch.autograd.grad(self.last_unweighted_pde_tensor, last_param, retain_graph=True)[0]
        grad_if = torch.autograd.grad(self.last_unweighted_if_tensor, last_param, retain_graph=True)[0]

        norm_pde = torch.mean(torch.abs(grad_pde))
        norm_if = torch.mean(torch.abs(grad_if))

        w_target = norm_pde / (norm_if + 1e-8)

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
