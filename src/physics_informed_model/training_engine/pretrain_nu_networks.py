import torch
import torch.optim as optim
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def pretrain_nu_networks(training_manager_instance, domains_data, epochs=None, lr=None):
    model = training_manager_instance.model
    if not hasattr(model, 'nu_networks') or len(model.nu_networks) == 0:
        return

    pinn = training_manager_instance.electro_magnetic_pinn
    t_cfg = getattr(pinn, 'training_config', None)
    epochs = epochs if epochs is not None else getattr(t_cfg, 'pretrain_nu_epochs', 300)
    lr = lr if lr is not None else getattr(t_cfg, 'pretrain_nu_lr', 1e-2)

    print(f"\n[*] Pre-training Mạng phụ Net_nu về hằng số ban đầu ({epochs} epochs, lr={lr})...")
    device = next(model.parameters()).device
    sim_domain = pinn.simulation_domain_engine_instance
    
    nu_params = []
    for net in model.nu_networks.values():
        nu_params.extend(net.parameters())
        
    opt_nu = optim.Adam(nu_params, lr=lr)
    domains_dict = domains_data.get("domains", domains_data)
    
    for ep in range(epochs):
        opt_nu.zero_grad()
        loss_pre = torch.tensor(0.0, device=device)
        
        for key, data in domains_dict.items():
            domain = data["domain_object"]
            net_key = resolve_domain_net_key(domain, model)
            if net_key in model.nu_networks:
                pts = data["interior_points"].to(device)
                
                if pts.shape[0] > 0:
                    nu_pred_star = model.forward_nu_network(net_key, pts)
                    nu_target_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                    nu_target_star = nu_target_phys / sim_domain.nu0
                    
                    loss_domain = torch.mean((nu_pred_star - nu_target_star) ** 2)
                    loss_pre = loss_pre + loss_domain
                    
        if loss_pre.requires_grad:
            loss_pre.backward()
            opt_nu.step()
