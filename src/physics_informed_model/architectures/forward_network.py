import torch
import numpy as np

def forward_network(self, network_key, xy):
    if network_key not in self.networks:
        raise ValueError(f"Lỗi: Không tìm thấy mạng con '{network_key}' trong XPINNArchitecture.")
        
    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == network_key:
            target_domain = domain
            break
        if f"domain_{domain.domain_id}" == network_key:
            target_domain = domain
            break
            
    if target_domain is None:
        raise ValueError(f"Lỗi: Không tìm thấy domain tương ứng với network_key '{network_key}' trong domains_list.")

    config = target_domain.network_config
    
    if config.fourier_features:
        B_matrix = getattr(self, f'B_matrix_{network_key}')
        if B_matrix.device != xy.device:
            B_matrix = B_matrix.to(xy.device)
        projected = (2.0 * np.pi * xy_normalized) @ B_matrix.T
        x_input = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
    else:
        x_input = xy_normalized
        
    raw_output = self.networks[network_key](x_input)
    
    if network_key == 'background':
        boundary_obj = sim_domain.boundary
        enforce_method = getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor')
        pot_val_star = float(getattr(boundary_obj.physics_config, 'potential_value', 0.0)) / max(sim_domain.A0, 1e-12)
        if enforce_method == 'hard_factor':
            A_z_star = raw_output * self.boundary_factor(xy) + pot_val_star
        else:
            A_z_star = raw_output
    else:
        A_z_star = raw_output
        
    return A_z_star
