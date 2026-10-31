import torch
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
            is_nonlinear_domain = getattr(domain.physics_config, 'is_nonlinear', False) or \
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
                    f"[*] XPINNArchitecture: Đã khởi tạo Mạng phụ $Net_{{\\nu}}$ cho miền '{domain.domain_name}' ({net_key}) "
                    f"[{nu_config.hidden_layers} layers x {nu_config.hidden_neurons} neurons]."
                )
            
        self._initialize_weights()
