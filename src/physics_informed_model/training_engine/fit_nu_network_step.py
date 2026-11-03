import torch
import torch.optim as optim
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def fit_nu_network_step(training_manager_instance, domains_data, alpha_damping=None, epochs=None, lr=None):
    model = training_manager_instance.model
    if not hasattr(model, 'nu_networks') or len(model.nu_networks) == 0:
        return 0.0

    pinn = training_manager_instance.electro_magnetic_pinn
    t_cfg = getattr(pinn, 'training_config', None)
    alpha_damping = alpha_damping if alpha_damping is not None else getattr(t_cfg, 'fpi_alpha_damping', 1.0)
    epochs = epochs if epochs is not None else getattr(t_cfg, 'fit_nu_epochs', 400)
    lr = lr if lr is not None else getattr(t_cfg, 'fit_nu_lr', 3e-3)

    device = next(model.parameters()).device
    sim_domain = pinn.simulation_domain_engine_instance
    domains_dict = domains_data.get("domains", domains_data)
    
    targets_dict = {}
    max_nu_diff = 0.0

    for key, data in domains_dict.items():
        domain = data["domain_object"]
        net_key = resolve_domain_net_key(domain, model)
        if net_key in model.nu_networks:
            if not getattr(domain.physics_config, 'is_nonlinear', False):
                continue

            pts = data["interior_points"].to(device)
            if pts.shape[0] > 0:
                pts_grad = pts.detach().clone().requires_grad_(True)
                
                # 1. Tính từ thông B(x,y) thực tế từ Net_Az
                Az_star = model.forward_network(net_key, pts_grad)
                grad_Az = torch.autograd.grad(
                    outputs=Az_star, inputs=pts_grad,
                    grad_outputs=torch.ones_like(Az_star), create_graph=True
                )[0]
                Bx_star = grad_Az[:, 1:2] * sim_domain.L0
                By_star = -grad_Az[:, 0:1] * sim_domain.L0
                B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * sim_domain.B0
                
                # 2. Đánh giá độ từ kháng phi tuyến thực tế từ đường cong B-H
                nu_BH_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys, is_nonlinear=True)
                nu_BH_star = nu_BH_phys / sim_domain.nu0

                # 3. Lấy nu hiện tại từ Net_nu
                nu_old_star = model.forward_nu_network(net_key, pts)
                
                # 4. Giữ nguyên Đồ thị Đạo hàm (Graph) cho quá trình giảm chấn FPI
                nu_target_star = (1.0 - alpha_damping) * nu_old_star + alpha_damping * nu_BH_star
                
                diff_val = torch.max(torch.abs(nu_target_star.detach() - nu_old_star.detach())).item()
                max_nu_diff = max(max_nu_diff, diff_val)
                targets_dict[net_key] = (pts, nu_target_star.detach())

    if not targets_dict:
        return 0.0

    nu_params = []
    for k in targets_dict.keys():
        nu_params.extend(model.nu_networks[k].parameters())
    opt_nu = optim.Adam(nu_params, lr=lr)

    for _ in range(epochs):
        opt_nu.zero_grad()
        total_fit_loss = torch.tensor(0.0, device=device)
        for k, (pts, nu_target_star) in targets_dict.items():
            nu_pred_star = model.forward_nu_network(k, pts)
            total_fit_loss = total_fit_loss + torch.mean((nu_pred_star - nu_target_star) ** 2)
        total_fit_loss.backward()
        opt_nu.step()

    return max_nu_diff
