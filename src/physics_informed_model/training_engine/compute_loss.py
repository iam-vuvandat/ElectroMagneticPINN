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
    total_bc_tensor = zero_tensor

    az_if_sum = 0.0
    bn_if_sum = 0.0
    ht_if_sum = 0.0
    weighted_if_sum = 0.0

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    pinn = training_manager_instance.electro_magnetic_pinn
    sim_domain = pinn.simulation_domain_engine_instance
    bg_domain = sim_domain.background_domain

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

    boundary_obj = getattr(sim_domain, 'boundary', None)
    enforce_method = getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor') if boundary_obj else 'hard_factor'
    bc_pts = domains_data.get("boundary_points")

    if enforce_method == 'soft_loss' and bc_pts is not None and bc_pts.shape[0] > 0:
        bc_pts_dev = bc_pts.to(device)
        pot_val_star = float(getattr(boundary_obj.physics_config, 'potential_value', 0.0)) / max(sim_domain.A0, 1e-12)
        phys_props = sim_domain.evaluate_global_physical_properties(bc_pts_dev)
        mat_id = phys_props["material_classification"].view(-1)

        sq_err_sum = zero_tensor
        total_bc_count = 0
        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, model)
            mask = (mat_id == float(d_id))
            if mask.any():
                sub_pts = bc_pts_dev[mask]
                Az_bc_pred = model.forward_network(net_key, sub_pts)
                sq_err_sum = sq_err_sum + torch.sum((Az_bc_pred - pot_val_star) ** 2)
                total_bc_count += sub_pts.shape[0]

        if total_bc_count > 0:
            raw_bc_loss = sq_err_sum / float(total_bc_count)
            w_bc = float(getattr(boundary_obj.training_config, 'boundary_loss_weight', 1.0))
            total_bc_tensor = w_bc * raw_bc_loss

    balancer = getattr(training_manager_instance, 'relobralo_balancer', None)
    if balancer is not None:
        raw_losses = {
            "loss_pde": total_pde_tensor,
            "loss_a": total_az_tensor,
            "loss_bn": total_bn_tensor,
            "loss_ht": total_ht_tensor
        }
        if enforce_method == 'soft_loss':
            raw_losses["loss_bc"] = total_bc_tensor

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
        if enforce_method == 'soft_loss':
            total_loss = total_loss + lambdas.get("loss_bc", 1.0) * total_bc_tensor
    else:
        lambdas = {
            "loss_pde": 1.0,
            "loss_a": 1.0,
            "loss_bn": 1.0,
            "loss_ht": 1.0
        }
        if enforce_method == 'soft_loss':
            lambdas["loss_bc"] = 1.0
        total_loss = total_pde_tensor + total_az_tensor + total_bn_tensor + total_ht_tensor + total_bc_tensor

    training_manager_instance.last_loss_breakdown = {
        "pde_loss": total_pde_tensor.item(),
        "interface_Az": az_if_sum,
        "interface_Bn": bn_if_sum,
        "interface_Ht": ht_if_sum,
        "boundary_loss": total_bc_tensor.item(),
        "weighted_interface": weighted_if_sum,
        "relobralo_lambdas": dict(lambdas),
        "total": total_loss.item()
    }

    return total_loss