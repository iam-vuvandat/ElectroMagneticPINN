import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY
from training_visualizer.resolve_domain_net_key import resolve_domain_net_key

def save_frame(visualizer_instance, epoch, loss_value, domains_data):
    visualizer_instance.loss_history.append(loss_value)
    visualizer_instance.epoch_history.append(epoch)
    
    model = visualizer_instance.electro_magnetic_pinn.pinn_architecture_instance
    pde_evaluator = visualizer_instance.electro_magnetic_pinn.maxwell_pde_loss_instance
    sim_domain = visualizer_instance.electro_magnetic_pinn.simulation_domain_engine_instance
    vis_cfg = visualizer_instance.electro_magnetic_pinn.visualization_config

    model.eval()
    computation_device = next(model.parameters()).device
    eval_points = visualizer_instance.eval_points_tensor.to(computation_device).detach().clone().requires_grad_(True)
    
    phys_props = sim_domain.evaluate_global_physical_properties(eval_points)
    mat_id = phys_props["material_classification"]
    
    A_z_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    B_x_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    B_y_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    nu_phys_all = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    residual_total = torch.zeros((eval_points.shape[0], 1), device=computation_device)

    for domain in sim_domain.domains_list:
        d_id = domain.domain_id
        net_key = resolve_domain_net_key(domain, model)
        mask_bool = (mat_id == float(d_id))
        if mask_bool.any():
            Az_sub_star = model.forward_network(net_key, eval_points)
            grad_sub_star = torch.autograd.grad(
                outputs=Az_sub_star, inputs=eval_points,
                grad_outputs=torch.ones_like(Az_sub_star), create_graph=True
            )[0]
            Az_sub_phys = Az_sub_star * sim_domain.A0
            Bx_sub_phys = grad_sub_star[:, 1:2] * sim_domain.A0
            By_sub_phys = -grad_sub_star[:, 0:1] * sim_domain.A0

            if getattr(domain.physics_config, 'is_nonlinear', False):
                if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                    with torch.no_grad():
                        nu_star_pred = model.forward_nu_network(net_key, eval_points)
                        nu_dom = nu_star_pred * sim_domain.nu0
                else:
                    B_mag_sub = torch.sqrt(Bx_sub_phys**2 + By_sub_phys**2 + 1e-12)
                    nu_dom = domain.evaluate_reluctivity(eval_points, B_magnitude_tensor=B_mag_sub, is_nonlinear=True).to(computation_device)
            else:
                nu_dom = domain.evaluate_reluctivity(eval_points, is_nonlinear=False).to(computation_device)

            hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
            hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
            jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

            res_pde = pde_evaluator.compute_linear_pde_residual(
                eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom
            )

            A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
            B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
            B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
            nu_phys_all = torch.where(mask_bool, nu_dom.detach(), nu_phys_all)
            residual_total = torch.where(mask_bool, torch.abs(res_pde.detach()), residual_total)
    
    A_z_numpy = A_z_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_x_numpy = B_x_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_y_numpy = B_y_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
    mu_r_numpy = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys_all, min=1e-6)).cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    residual_numpy = residual_total.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    
    fig, axs = plt.subplots(2, 4, figsize=(23, 10))
    n_levels = getattr(vis_cfg, 'contour_levels', 60)

    # --- SUBPLOT 1: axs[0, 0] ---
    ax_pts = axs[0, 0]
    ax_pts.set_title("Categorized Collocation Points")
    
    # Bảng màu đậm (Dark/Deep palette) có độ tương phản cao
    domain_colors = ['#1a365d', '#742a2a', '#1c4532', '#4a5568', '#44337a', '#7b341e']
    pair_colors = ['#800000', '#004080', '#004d40', '#4a148c', '#bf360c']

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})

    for idx, (key, d) in enumerate(domains_dict.items()):
        c_dom = domain_colors[idx % len(domain_colors)]
        int_pts = d.get("interior_points")
        if int_pts is not None and int_pts.shape[0] > 0:
            pts_np = int_pts.detach().cpu().numpy()
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=0.5, color=c_dom, marker='o', linewidths=0, alpha=0.55)

    for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
        if_pts = pair_data.get("interface_points")
        if if_pts is not None and if_pts.shape[0] > 0:
            pts_np = if_pts.detach().cpu().numpy()
            c_pair = pair_colors[p_idx % len(pair_colors)]
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=1.0, color=c_pair, marker='o', linewidths=0, alpha=0.95)

    # Đã loại bỏ ax_pts.legend()
    ax_pts.set_xlim(visualizer_instance.x_bounds)
    ax_pts.set_ylim(visualizer_instance.y_bounds)
    ax_pts.set_aspect('equal')
    
    # --- SUBPLOT 2: axs[0, 1] ---
    res_min = vis_cfg.res_vmin if vis_cfg.res_vmin is not None else float(np.min(residual_numpy))
    res_max = vis_cfg.res_vmax if vis_cfg.res_vmax is not None else float(np.max(residual_numpy))

    if getattr(vis_cfg, 'res_scale', 'log') == 'log':
        res_min_clean = max(res_min, 1e-6)
        res_max_clean = max(res_max, res_min_clean * 10.0)
        res_abs = np.clip(np.abs(residual_numpy), res_min_clean, res_max_clean)
        norm_res = LogNorm(vmin=res_min_clean, vmax=res_max_clean)
        levels_res = np.logspace(np.log10(res_min_clean), np.log10(res_max_clean), n_levels)
    else:
        res_abs = residual_numpy
        norm_res = Normalize(vmin=res_min, vmax=res_max)
        levels_res = np.linspace(res_min, res_max, n_levels)

    contour_res = axs[0, 1].contourf(visualizer_instance.X_grid, visualizer_instance.Y_grid, res_abs, levels=levels_res, cmap=vis_cfg.res_cmap, norm=norm_res)
    fig.colorbar(contour_res, ax=axs[0, 1])
    axs[0, 1].set_title("PDE Residual Error")
    axs[0, 1].set_aspect('equal')

    # --- SUBPLOT 3: axs[0, 2] ---
    mur_min = vis_cfg.mur_vmin if vis_cfg.mur_vmin is not None else 0.0
    mur_max = vis_cfg.mur_vmax if vis_cfg.mur_vmax is not None else 3000.0
    mu_r_clamped = np.clip(mu_r_numpy, mur_min, mur_max)

    if getattr(vis_cfg, 'mur_scale', 'linear') == 'log':
        mur_min_clean = max(mur_min, 1e-3)
        mur_max_clean = max(mur_max, mur_min_clean * 2.0)
        norm_mur = LogNorm(vmin=mur_min_clean, vmax=mur_max_clean)
        levels_mur = np.logspace(np.log10(mur_min_clean), np.log10(mur_max_clean), n_levels)
        lbl_mur = r"$\mu_r$ (Log Scale)"
    else:
        norm_mur = Normalize(vmin=mur_min, vmax=mur_max)
        levels_mur = np.linspace(mur_min, mur_max, n_levels)
        lbl_mur = r"$\mu_r$ (Linear Scale: 0 - 3000)"

    contour_mur = axs[0, 2].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, mu_r_clamped, levels=levels_mur,
        cmap=vis_cfg.mur_cmap, norm=norm_mur
    )
    fig.colorbar(contour_mur, ax=axs[0, 2], label=lbl_mur)
    axs[0, 2].set_title(r"Permeability Distribution ($\mu_r$)")
    axs[0, 2].set_aspect('equal')
    
    # --- SUBPLOT 4: axs[0, 3] ---
    axs[0, 3].plot(visualizer_instance.epoch_history, visualizer_instance.loss_history, 'b-')
    axs[0, 3].set_yscale('log')
    axs[0, 3].set_title("Loss Optimization")
    axs[0, 3].set_xlabel("Total Optimization Step")
    axs[0, 3].set_ylabel("Loss")
    axs[0, 3].grid(True, which="both", ls="-", alpha=0.2)
    
    # --- SUBPLOT 5: axs[1, 0] ---
    az_min = vis_cfg.az_vmin if vis_cfg.az_vmin is not None else float(np.min(A_z_numpy))
    az_max = vis_cfg.az_vmax if vis_cfg.az_vmax is not None else float(np.max(A_z_numpy))
    levels_az = np.linspace(az_min, az_max, n_levels) if az_min != az_max else n_levels

    contour_az = axs[1, 0].contourf(visualizer_instance.X_grid, visualizer_instance.Y_grid, A_z_numpy, levels=levels_az, cmap=vis_cfg.az_cmap)
    fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
    axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
    axs[1, 0].set_aspect('equal')

    # --- SUBPLOT 6: axs[1, 1] ---
    b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
    b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else 1.8
    b_mag_clamped = np.clip(B_mag_numpy, b_min, b_max)
    levels_b = np.linspace(b_min, b_max, n_levels)

    contour_b = axs[1, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=vis_cfg.b_mag_cmap, vmin=b_min, vmax=b_max
    )
    fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
    axs[1, 1].set_title(f"Flux Density Magnitude ($|B|$ fixed [{b_min}-{b_max}T])")
    axs[1, 1].set_aspect('equal')

    # --- SUBPLOT 7: axs[1, 2] ---
    contour_b_bg = axs[1, 2].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=vis_cfg.b_mag_cmap, alpha=0.3, vmin=b_min, vmax=b_max
    )
    fig.colorbar(contour_b_bg, ax=axs[1, 2], label="|B| (T)")
    step = max(1, visualizer_instance.resolution // 20)
    B_sub = np.maximum(B_mag_numpy[::step, ::step], 1e-12)
    axs[1, 2].quiver(
        visualizer_instance.X_grid[::step, ::step], visualizer_instance.Y_grid[::step, ::step], 
        B_x_numpy[::step, ::step] / B_sub, B_y_numpy[::step, ::step] / B_sub, 
        scale=30.0, color='black', pivot='mid'
    )
    axs[1, 2].set_title("Flux Density Vectors (B)")
    axs[1, 2].set_aspect('equal')

    # --- SUBPLOT 8: axs[1, 3] ---
    mat_id_numpy = mat_id.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    im_mat = axs[1, 3].imshow(mat_id_numpy, extent=[visualizer_instance.x_bounds[0], visualizer_instance.x_bounds[1], visualizer_instance.y_bounds[0], visualizer_instance.y_bounds[1]], origin='lower', interpolation='nearest', cmap='tab10')
    fig.colorbar(im_mat, ax=axs[1, 3], label="Domain ID")
    axs[1, 3].set_title("Material Classification")
    axs[1, 3].set_aspect('equal')
    
    fig.suptitle(f"XPINN Training Process - Step {epoch}", fontsize=16)
    plt.tight_layout()
    
    frame_name = os.path.join(visualizer_instance.output_directory, f"frame_{len(visualizer_instance.frame_paths) + 1:05d}.png")
    plt.savefig(frame_name, dpi=100)
    plt.close(fig)
    
    visualizer_instance.frame_paths.append(frame_name)
    model.train()
