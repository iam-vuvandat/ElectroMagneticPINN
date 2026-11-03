import time
import torch
import torch.optim as optim

from training_engine.resolve_domain_net_key import resolve_domain_net_key
from training_engine.pretrain_nu_networks import pretrain_nu_networks
from training_engine.fit_nu_network_step import fit_nu_network_step
from training_engine.compute_loss import compute_loss
from training_engine.train_adam import train_adam
from training_engine.train_lbfgs import train_lbfgs


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
        
        # Đưa cả tham số của Net_Az và Net_nu vào Adam để tối ưu thống nhất khi cần
        # Chỉ tối ưu hóa tham số của Net_Az trong Adam. Net_nu được cập nhật riêng qua bước FPI (fit_nu_network_step)
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
        return pretrain_nu_networks(self, domains_data, epochs=epochs, lr=lr)

    def fit_nu_network_step(self, domains_data, alpha_damping=None, epochs=None, lr=None):
        return fit_nu_network_step(self, domains_data, alpha_damping=alpha_damping, epochs=epochs, lr=lr)

    def compute_loss(self, domains_data):
        return compute_loss(self, domains_data)

    def train_adam(self, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
        return train_adam(self, epochs, domains_data, dynamic_resampling=dynamic_resampling, resample_fn=resample_fn, resample_interval=resample_interval)

    def train_lbfgs(self, max_iterations, domains_data):
        return train_lbfgs(self, max_iterations, domains_data)
