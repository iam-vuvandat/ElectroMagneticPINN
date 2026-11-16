# src/physics_informed_model/electro_magnetic_pinn_engine/evaluate_fields.py
import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def evaluate_fields(pinn_instance, points_tensor, return_numpy=False):
    pinn_arch = pinn_instance.pinn_architecture_instance
    sim_domain = pinn_instance.simulation_domain_engine_instance
    device = points_tensor.device

    with torch.no_grad():
        phys_props = sim_domain.evaluate_global_physical_properties(points_tensor)
        mat_id = phys_props["material_classification"]

    A_z_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
    B_x_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
    B_y_phys = torch.zeros((points_tensor.shape[0], 1), device=device)

    for domain in sim_domain.domains_list:
        d_id = domain.domain_id
        net_key = resolve_domain_net_key(domain, pinn_arch)
        mask_bool = (mat_id == float(d_id)).view(-1)
        
        if mask_bool.any():
            sub_pts = points_tensor[mask_bool].detach().clone().requires_grad_(True)
            Az_star = pinn_arch.forward_network(net_key, sub_pts)
            grad_star = torch.autograd.grad(
                outputs=Az_star, inputs=sub_pts,
                grad_outputs=torch.ones_like(Az_star), create_graph=False
            )[0]

            A_z_phys[mask_bool] = (Az_star * pinn_instance.A0).detach()
            B_x_phys[mask_bool] = (grad_star[:, 1:2] * pinn_instance.A0).detach()
            B_y_phys[mask_bool] = (-grad_star[:, 0:1] * pinn_instance.A0).detach()

    if return_numpy:
        return (
            A_z_phys.cpu().numpy(),
            B_x_phys.cpu().numpy(),
            B_y_phys.cpu().numpy()
        )

    return A_z_phys, B_x_phys, B_y_phys