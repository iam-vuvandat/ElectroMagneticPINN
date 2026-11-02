import torch
import torch.nn as nn
import numpy as np

def forward_network(self, network_key, xy):
    actual_key = network_key
    if actual_key not in self.networks:
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        for domain in sim_domain.domains_list:
            d_name = getattr(domain, 'domain_name', '')
            d_id = getattr(domain, 'domain_id', None)
            if d_name == network_key or f"domain_{d_id}" == network_key:
                if f"domain_{d_id}" in self.networks:
                    actual_key = f"domain_{d_id}"
                    break
                elif d_name in self.networks:
                    actual_key = d_name
                    break

    if actual_key not in self.networks:
        raise ValueError(f"Lỗi: Không tìm thấy mạng con '{network_key}' trong XPINNArchitecture.")
        
    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == actual_key or f"domain_{domain.domain_id}" == actual_key:
            target_domain = domain
            break
            
    if target_domain is None:
        raise ValueError(f"Lỗi: Không tìm thấy domain tương ứng với network_key '{actual_key}'.")

    config = target_domain.network_config
    
    if config.fourier_features:
        B_matrix = getattr(self, f'B_matrix_{actual_key}', None)
        if B_matrix is not None:
            if B_matrix.device != xy.device:
                B_matrix = B_matrix.to(xy.device)
            projected = (2.0 * np.pi * xy_normalized) @ B_matrix.T
            x_input = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        else:
            x_input = xy_normalized
    else:
        x_input = xy_normalized
        
    raw_output = self.networks[actual_key](x_input)
    
    if actual_key == 'background' or getattr(target_domain, 'domain_name', '') == 'background':
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


def forward_nu_network(self, network_key, xy):
    """
    Tra cứu độ từ kháng nu* = nu / nu0 từ Mạng phụ Net_nu.
    Hỗ trợ đối chiếu tự động giữa tên domain và domain_id.
    """
    actual_key = network_key
    if actual_key not in self.nu_networks:
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        for domain in sim_domain.domains_list:
            d_name = getattr(domain, 'domain_name', '')
            d_id = getattr(domain, 'domain_id', None)
            if d_name == network_key or f"domain_{d_id}" == network_key:
                if f"domain_{d_id}" in self.nu_networks:
                    actual_key = f"domain_{d_id}"
                    break
                elif d_name in self.nu_networks:
                    actual_key = d_name
                    break

    if actual_key not in self.nu_networks:
        return None

    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0
    
    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == actual_key or f"domain_{domain.domain_id}" == actual_key:
            target_domain = domain
            break

    nu_config = getattr(target_domain, 'nu_network_config', None) if target_domain else None

    if nu_config and getattr(nu_config, 'fourier_features', False):
        B_matrix_nu = getattr(self, f'B_matrix_nu_{actual_key}', None)
        if B_matrix_nu is not None:
            if B_matrix_nu.device != xy.device:
                B_matrix_nu = B_matrix_nu.to(xy.device)
            projected = (2.0 * np.pi * xy_normalized) @ B_matrix_nu.T
            x_input = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        else:
            x_input = xy_normalized
    else:
        x_input = xy_normalized

    raw_nu = self.nu_networks[actual_key](x_input)
    # Softplus + 1e-6 đảm bảo nu* luôn dương
    nu_star = torch.nn.functional.softplus(raw_nu) + 1e-6
    return nu_star
