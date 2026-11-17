import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def evaluate_fields(pinn_instance, points_tensor, return_numpy=False):
    pinn_arch = pinn_instance.pinn_architecture_instance
    sim_domain = pinn_instance.simulation_domain_engine_instance
    device = points_tensor.device

    phys_props = sim_domain.evaluate_global_physical_properties(points_tensor)
    mat_id = phys_props["material_classification"]

    B_x_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
    B_y_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
    H_x_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
    H_y_phys = torch.zeros((points_tensor.shape[0], 1), device=device)

    with torch.no_grad():
        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, pinn_arch)
            mask_bool = (mat_id == float(d_id))

            if mask_bool.any():
                field_star = pinn_arch.forward_network(net_key, points_tensor)
                Bx_phys = field_star[:, 0:1] * sim_domain.B0
                By_phys = field_star[:, 1:2] * sim_domain.B0

                if getattr(domain.physics_config, 'is_nonlinear', False):
                    if hasattr(pinn_arch, 'nu_networks') and net_key in pinn_arch.nu_networks:
                        nu_star_pred = pinn_arch.forward_nu_network(net_key, points_tensor)
                        nu_phys = nu_star_pred * sim_domain.nu0
                    else:
                        B_mag_sub = torch.sqrt(Bx_phys**2 + By_phys**2 + 1e-12)
                        nu_phys = domain.evaluate_reluctivity(points_tensor, B_magnitude_tensor=B_mag_sub, is_nonlinear=True).to(device)
                else:
                    nu_phys = domain.evaluate_reluctivity(points_tensor, is_nonlinear=False).to(device)

                hx_c, hy_c = domain.evaluate_magnetization_vector(points_tensor)
                hx_c, hy_c = hx_c.to(device), hy_c.to(device)
                Hx_phys = nu_phys * Bx_phys - hx_c
                Hy_phys = nu_phys * By_phys - hy_c

                B_x_phys = torch.where(mask_bool, Bx_phys, B_x_phys)
                B_y_phys = torch.where(mask_bool, By_phys, B_y_phys)
                H_x_phys = torch.where(mask_bool, Hx_phys, H_x_phys)
                H_y_phys = torch.where(mask_bool, Hy_phys, H_y_phys)

    if return_numpy:
        return (
            B_x_phys.detach().cpu().numpy(),
            B_y_phys.detach().cpu().numpy(),
            H_x_phys.detach().cpu().numpy(),
            H_y_phys.detach().cpu().numpy()
        )

    return B_x_phys.detach(), B_y_phys.detach(), H_x_phys.detach(), H_y_phys.detach()