import torch
import torch.nn as nn
import numpy as np

from .build_single_network import _build_single_network
from .initialize_weights import _initialize_weights
from .boundary_factor import boundary_factor
from .forward_network import forward_network
from .forward import forward

class XPINNArchitecture(nn.Module):
    _build_single_network = _build_single_network
    _initialize_weights = _initialize_weights
    boundary_factor = boundary_factor
    forward_network = forward_network
    forward = forward

    def __init__(self, electro_magnetic_pinn, input_dim=2, output_dim=1, domain_scale=None):
        super().__init__()
        
        self.electro_magnetic_pinn = electro_magnetic_pinn
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        self.domain_scale = domain_scale if domain_scale is not None else sim_domain.L0
        
        self.networks = nn.ModuleDict()
        self.domain_count = sim_domain.get_domain_count()
        domains_list = sim_domain.domains_list
        
        for domain in domains_list:
            if getattr(domain, 'domain_name', '') == 'background':
                net_key = 'background'
            else:
                net_key = f'domain_{domain.domain_id}'
                
            config = domain.network_config
            if config.fourier_features:
                actual_input_dim = config.fourier_mapping_size * 2
                B_mat = torch.randn((config.fourier_mapping_size, self.input_dim)) * config.fourier_sigma
                self.register_buffer(f'B_matrix_{net_key}', B_mat)
            else:
                actual_input_dim = self.input_dim
                
            sub_net = self._build_single_network(actual_input_dim, config)
            self.networks[net_key] = sub_net
            
        self._initialize_weights()
