import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def evaluate_fields(pinn_instance, points_tensor, return_numpy=False):
    pinn_arch = pinn_instance.pinn_architecture_instance
    sim_domain = pinn_instance.simulation_domain_engine_instance
    device = points_tensor.device

    phys_props = sim_domain.evaluate_global_physical_properties(points_tensor)
    mat_id = phys_props["material_classification"]

    A_z_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
    B_x_phys = torch.zeros((points_tensor.shape[0], 1), device=device)
    B_y_phys = torch.zeros((points_tensor.shape[0], 1), device=device)

    for domain in sim_domain.domains_list:
        d_id = domain.domain_id
        net_key = resolve_domain_net_key(domain, pinn_arch)
        mask_bool = (mat_id == float(d_id))
        
        if mask_bool.any():
            Az_star = pinn_arch.forward_network(net_key, points_tensor)
            grad_star = torch.autograd.grad(
                outputs=Az_star, inputs=points_tensor,
                grad_outputs=torch.ones_like(Az_star), create_graph=False
            )[0]

            Az_phys = Az_star * pinn_instance.A0
            Bx_phys = grad_star[:, 1:2] * pinn_instance.A0
            By_phys = -grad_star[:, 0:1] * pinn_instance.A0

            A_z_phys = torch.where(mask_bool, Az_phys, A_z_phys)
            B_x_phys = torch.where(mask_bool, Bx_phys, B_x_phys)
            B_y_phys = torch.where(mask_bool, By_phys, B_y_phys)

    if return_numpy:
        return (
            A_z_phys.detach().cpu().numpy(),
            B_x_phys.detach().cpu().numpy(),
            B_y_phys.detach().cpu().numpy()
        )

    return A_z_phys.detach(), B_x_phys.detach(), B_y_phys.detach()
