import torch
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
        raise ValueError(f"Error: Sub-network '{network_key}' not found in XPINNArchitecture.")

    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    xy_normalized = xy / sim_domain.L0

    target_domain = None
    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') == actual_key or f"domain_{domain.domain_id}" == actual_key:
            target_domain = domain
            break

    if target_domain is None:
        raise ValueError(f"Error: Domain corresponding to network_key '{actual_key}' not found.")

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
        if enforce_method == 'hard_factor':
            x_min, x_max = boundary_obj.geometry_config.x_boundaries_tuple
            y_min, y_max = boundary_obj.geometry_config.y_boundaries_tuple
            x_norm = 2.0 * (xy[:, 0:1] - x_min) / (x_max - x_min) - 1.0
            y_norm = 2.0 * (xy[:, 1:2] - y_min) / (y_max - y_min) - 1.0
            bx_star = raw_output[:, 0:1] * (1.0 - x_norm ** 2)
            by_star = raw_output[:, 1:2] * (1.0 - y_norm ** 2)
            return torch.cat([bx_star, by_star], dim=-1)

    return raw_output


def forward_nu_network(self, network_key, xy):
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
    nu_star = torch.nn.functional.softplus(raw_nu) + 1e-6
    return nu_star