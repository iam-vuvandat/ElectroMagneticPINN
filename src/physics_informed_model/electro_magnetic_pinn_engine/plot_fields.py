import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def plot_fields(pinn_instance, resolution=180, save_path=None, show=False, figsize=(12, 10)):
    sim_domain = pinn_instance.simulation_domain_engine_instance
    vis_cfg = pinn_instance.visualization_config
    pinn_arch = pinn_instance.pinn_architecture_instance

    xb, yb = sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple
    x_coords = np.linspace(xb[0], xb[1], resolution)
    y_coords = np.linspace(yb[0], yb[1], resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    device = pinn_instance.computation_device
    eval_pts = torch.tensor(
        np.column_stack((X_grid.ravel(), Y_grid.ravel())),
        dtype=torch.float32, device=device
    ).requires_grad_(True)
    
    A_z_np, B_x_np, B_y_np = pinn_instance.evaluate_fields(eval_pts, return_numpy=True)

    A_z_grid = A_z_np.reshape(resolution, resolution)
    B_x_grid = B_x_np.reshape(resolution, resolution)
    B_y_grid = B_y_np.reshape(resolution, resolution)
    B_mag_grid = np.sqrt(B_x_grid ** 2 + B_y_grid ** 2)

    props = sim_domain.evaluate_global_physical_properties(eval_pts)
    mat_ids = props["material_classification"].squeeze()
    
    mu_r_tensor = torch.ones_like(mat_ids, device=device)
    B_mag_pts = torch.tensor(B_mag_grid.ravel(), dtype=torch.float32, device=device).unsqueeze(1)

    for domain in sim_domain.domains_list:
        if getattr(domain, 'domain_name', '') != 'background':
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, pinn_arch)
            mask = (mat_ids == float(d_id))
            if mask.any():
                if getattr(domain.physics_config, 'is_nonlinear', False):
                    if hasattr(pinn_arch, 'nu_networks') and net_key in pinn_arch.nu_networks:
                        with torch.no_grad():
                            nu_star_pred = pinn_arch.forward_nu_network(net_key, eval_pts[mask])
                            mu_r_vals = (1.0 / torch.clamp(nu_star_pred, min=1e-6)).view(-1)
                            mu_r_tensor[mask] = mu_r_vals
                    else:
                        nu_phys = domain.evaluate_reluctivity(eval_pts[mask], B_magnitude_tensor=B_mag_pts[mask], is_nonlinear=True)
                        mu_r_vals = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys, min=1e-6)).view(-1)
                        mu_r_tensor[mask] = mu_r_vals
                else:
                    mu_val = float(getattr(domain.physics_config, 'relative_permeability', 1.0))
                    mu_r_tensor[mask] = mu_val

    mu_r_grid = mu_r_tensor.detach().cpu().numpy().reshape(resolution, resolution)

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    n_levels = getattr(vis_cfg, 'contour_levels', 60)
    
    # --- 1. A_z Plot ---
    az_vmin = vis_cfg.az_vmin if vis_cfg.az_vmin is not None else float(np.min(A_z_grid))
    az_vmax = vis_cfg.az_vmax if vis_cfg.az_vmax is not None else float(np.max(A_z_grid))
    levels_az = np.linspace(az_vmin, az_vmax, n_levels) if az_vmin != az_vmax else n_levels
    c0 = axs[0, 0].contourf(X_grid, Y_grid, A_z_grid, levels=levels_az, cmap=vis_cfg.az_cmap)
    fig.colorbar(c0, ax=axs[0, 0], label=r"$A_z$ (Wb/m)")
    axs[0, 0].set_title(r"Magnetic Vector Potential ($A_z$)")
    axs[0, 0].set_aspect('equal')

    # --- 2. |B| Magnitude Plot ---
    b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
    b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else 1.8
    b_mag_clamped = np.clip(B_mag_grid, b_min, b_max)
    levels_b = np.linspace(b_min, b_max, n_levels)
    c1 = axs[0, 1].contourf(X_grid, Y_grid, b_mag_clamped, levels=levels_b, cmap=vis_cfg.b_mag_cmap, vmin=b_min, vmax=b_max)
    fig.colorbar(c1, ax=axs[0, 1], label=r"$|B|$ (T)")
    axs[0, 1].set_title(r"Magnetic Flux Density Magnitude ($|B|$)")
    axs[0, 1].set_aspect('equal')

    # --- 3. Relative Permeability \mu_r Plot ---
    v_min_mur = vis_cfg.mur_vmin if vis_cfg.mur_vmin is not None else 0.0
    v_max_mur = vis_cfg.mur_vmax if vis_cfg.mur_vmax is not None else 3000.0
    mu_r_clamped = np.clip(mu_r_grid, v_min_mur, v_max_mur)
    
    if getattr(vis_cfg, 'mur_scale', 'linear') == 'log':
        levels_mur = np.logspace(np.log10(max(v_min_mur, 1e-3)), np.log10(v_max_mur), n_levels)
        norm_mur = LogNorm(vmin=max(v_min_mur, 1.0), vmax=v_max_mur)
        lbl_mur = r"$\mu_r$ (Log Scale)"
    else:
        levels_mur = np.linspace(v_min_mur, v_max_mur, n_levels)
        norm_mur = Normalize(vmin=v_min_mur, vmax=v_max_mur)
        lbl_mur = r"$\mu_r$ (Linear Scale: 0 - 3000)"

    c2 = axs[1, 0].contourf(X_grid, Y_grid, mu_r_clamped, levels=levels_mur, cmap=vis_cfg.mur_cmap, norm=norm_mur)
    fig.colorbar(c2, ax=axs[1, 0], label=lbl_mur)
    axs[1, 0].set_title(r"Relative Permeability Distribution ($\mu_r$)")
    axs[1, 0].set_aspect('equal')

    # --- 4. B_y Component Plot ---
    by_vmin = vis_cfg.by_vmin if vis_cfg.by_vmin is not None else float(np.min(B_y_grid))
    by_vmax = vis_cfg.by_vmax if vis_cfg.by_vmax is not None else float(np.max(B_y_grid))
    levels_by = np.linspace(by_vmin, by_vmax, n_levels) if by_vmin != by_vmax else n_levels
    c3 = axs[1, 1].contourf(X_grid, Y_grid, B_y_grid, levels=levels_by, cmap=vis_cfg.by_cmap)
    fig.colorbar(c3, ax=axs[1, 1], label=r"$B_y$ (T)")
    axs[1, 1].set_title(r"Field Component ($B_y$)")
    axs[1, 1].set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axs
