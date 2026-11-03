import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def compute_loss(training_manager_instance, domains_data):
    model = training_manager_instance.model
    device = next(model.parameters()).device
    dummy_param = next(model.parameters())
    total_loss = (dummy_param * 0.0).sum()

    pde_loss_sum = 0.0
    az_if_sum = 0.0
    bn_if_sum = 0.0
    ht_if_sum = 0.0
    weighted_if_sum = 0.0

    unweighted_pde_tensor = (dummy_param * 0.0).sum()
    unweighted_if_tensor = (dummy_param * 0.0).sum()

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    pinn = training_manager_instance.electro_magnetic_pinn
    bg_domain = pinn.simulation_domain_engine_instance.background_domain
    t_cfg = getattr(pinn, 'training_config', None)

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
            unweighted_pde_tensor = unweighted_pde_tensor + loss_pde_domain

            w_pde = float(getattr(domain.training_config, 'pde_weight', 1.0))
            scaled_pde_loss = w_pde * loss_pde_domain
            total_loss = total_loss + scaled_pde_loss
            pde_loss_sum += scaled_pde_loss.item()

        outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
        if outer_pts.shape[0] > 0 and net_key != 'background':
            if not outer_pts.requires_grad:
                outer_pts = outer_pts.detach().clone().requires_grad_(True)
                
            w_if_loss, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
            )
            unweighted_if_tensor = unweighted_if_tensor + (val_l + bn_l + ht_l)
            scaled_if_loss = w_if_loss
            total_loss = total_loss + scaled_if_loss

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += scaled_if_loss.item()

    for pair_key, pair_data in shared_dict.items():
        pts = pair_data["interface_points"].to(device)
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)

            w_shared_loss, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts
            )
            unweighted_if_tensor = unweighted_if_tensor + (val_l + bn_l + ht_l)
            scaled_shared_loss = w_shared_loss
            total_loss = total_loss + scaled_shared_loss

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += scaled_shared_loss.item()

    # Lưu lại tensor loss phục vụ tính Grad-Norm
    training_manager_instance.last_unweighted_pde_tensor = unweighted_pde_tensor
    training_manager_instance.last_unweighted_if_tensor = unweighted_if_tensor

    # Nếu dùng phương pháp Uncertainty
    if getattr(t_cfg, 'enable_dynamic_weighting', False) and training_manager_instance.loss_weighter is not None:
        total_loss = training_manager_instance.loss_weighter(unweighted_pde_tensor, unweighted_if_tensor)

    training_manager_instance.last_loss_breakdown = {
        "pde_loss": pde_loss_sum,
        "interface_Az": az_if_sum,
        "interface_Bn": bn_if_sum,
        "interface_Ht": ht_if_sum,
        "weighted_interface": weighted_if_sum,
        "total": total_loss.item()
    }

    return total_loss
