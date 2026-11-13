import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key


def compute_loss(training_manager_instance, domains_data, update_relobralo=True, is_resampling_step=False):
    model = training_manager_instance.model
    device = next(model.parameters()).device
    dummy_param = next(model.parameters())
    zero_tensor = (dummy_param * 0.0).sum()

    total_pde_tensor = zero_tensor
    total_az_tensor = zero_tensor
    total_bn_tensor = zero_tensor
    total_ht_tensor = zero_tensor

    az_if_sum = 0.0
    bn_if_sum = 0.0
    ht_if_sum = 0.0
    weighted_if_sum = 0.0

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    pinn = training_manager_instance.electro_magnetic_pinn
    bg_domain = pinn.simulation_domain_engine_instance.background_domain

    for key, data in domains_dict.items():
        domain = data["domain_object"]
        net_key = resolve_domain_net_key(domain, model)
        pts = data["interior_points"].to(device)
        
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)
                
            A_z_star = model.forward_network(net_key, pts)
            
            if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                nu_star_pred = model.forward_nu_network(net_key, pts)
                nu_star_net = nu_star_pred if nu_star_pred is not None else None
            else:
                nu_star_net = None

            nu_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
            hx, hy = domain.evaluate_magnetization_vector(pts)
            hx, hy = hx.to(device), hy.to(device)
            jz = domain.evaluate_current_density(pts).to(device)
            
            res_pde = training_manager_instance.pde_evaluator.compute_linear_pde_residual(
                xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_star_net=nu_star_net
            )
            fading_w = training_manager_instance.pde_evaluator.compute_fading_weight(domain, pts)
            
            loss_pde_domain = torch.mean((res_pde ** 2) * fading_w)
            w_pde = float(getattr(domain.training_config, 'pde_weight', 1.0))

            scaled_pde_loss = w_pde * loss_pde_domain
            total_pde_tensor = total_pde_tensor + scaled_pde_loss

        outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
        if outer_pts.shape[0] > 0 and net_key != 'background':
            if not outer_pts.requires_grad:
                outer_pts = outer_pts.detach().clone().requires_grad_(True)
                
            _, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
            )
            w_if = float(getattr(domain.training_config, 'interface_weight', 1.0))
            w_az = float(getattr(domain.training_config, 'interface_potential_weight', 1.0))
            w_bn = float(getattr(domain.training_config, 'interface_bn_weight', 1.0))
            w_ht = float(getattr(domain.training_config, 'interface_ht_weight', 1.0))

            total_az_tensor = total_az_tensor + (w_if * w_az * val_l)
            total_bn_tensor = total_bn_tensor + (w_if * w_bn * bn_l)
            total_ht_tensor = total_ht_tensor + (w_if * w_ht * ht_l)

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += (w_if * (w_az * val_l + w_bn * bn_l + w_ht * ht_l)).item()

    for pair_key, pair_data in shared_dict.items():
        pts = pair_data["interface_points"].to(device)
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)

            _, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts
            )
            dom_a = pair_data["domain_a"]
            w_if = float(getattr(dom_a.training_config, 'interface_weight', 1.0))
            w_az = float(getattr(dom_a.training_config, 'interface_potential_weight', 1.0))
            w_bn = float(getattr(dom_a.training_config, 'interface_bn_weight', 1.0))
            w_ht = float(getattr(dom_a.training_config, 'interface_ht_weight', 1.0))

            total_az_tensor = total_az_tensor + (w_if * w_az * val_l)
            total_bn_tensor = total_bn_tensor + (w_if * w_bn * bn_l)
            total_ht_tensor = total_ht_tensor + (w_if * w_ht * ht_l)

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += (w_if * (w_az * val_l + w_bn * bn_l + w_ht * ht_l)).item()

    balancer = getattr(training_manager_instance, 'relobralo_balancer', None)
    if balancer is not None:
        raw_losses = {
            "loss_pde": total_pde_tensor,
            "loss_a": total_az_tensor,
            "loss_bn": total_bn_tensor,
            "loss_ht": total_ht_tensor
        }
        if update_relobralo:
            lambdas = balancer.update(raw_losses, is_resampling_step=is_resampling_step)
        else:
            lambdas = balancer.get_weights(raw_losses.keys())

        total_loss = (
            lambdas["loss_pde"] * total_pde_tensor
            + lambdas["loss_a"] * total_az_tensor
            + lambdas["loss_bn"] * total_bn_tensor
            + lambdas["loss_ht"] * total_ht_tensor
        )
    else:
        lambdas = {
            "loss_pde": 1.0,
            "loss_a": 1.0,
            "loss_bn": 1.0,
            "loss_ht": 1.0
        }
        total_loss = total_pde_tensor + total_az_tensor + total_bn_tensor + total_ht_tensor

    training_manager_instance.last_loss_breakdown = {
        "pde_loss": total_pde_tensor.item(),
        "interface_Az": az_if_sum,
        "interface_Bn": bn_if_sum,
        "interface_Ht": ht_if_sum,
        "weighted_interface": weighted_if_sum,
        "relobralo_lambdas": dict(lambdas),
        "total": total_loss.item()
    }

    return total_loss
