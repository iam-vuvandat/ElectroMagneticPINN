import os
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