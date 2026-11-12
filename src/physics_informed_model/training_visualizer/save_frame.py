import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY
from training_visualizer.resolve_domain_net_key import resolve_domain_net_key

ANSYS_FEM_HEX_COLORS = [
    '#0000FF', '#0036FF', '#006BFF', '#00A1FF', '#00D7FF', 
    '#00FFF2', '#00FFBC', '#00FF86', '#00FF51', '#00FF1B', 
    '#1BFF00', '#51FF00', '#86FF00', '#BCFF00', '#F2FF00', 
    '#FFD700', '#FFA100', '#FF6B00', '#FF3600', '#FF0000'
]

def save_frame(visualizer_instance, epoch, loss_value, domains_data):
    visualizer_instance.loss_history.append(loss_value)
    visualizer_instance.epoch_history.append(epoch)
    
    model = visualizer_instance.electro_magnetic_pinn.pinn_architecture_instance
    pde_evaluator = visualizer_instance.electro_magnetic_pinn.maxwell_pde_loss_instance
    interface_evaluator = visualizer_instance.electro_magnetic_pinn.interface_loss_instance
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

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    bg_domain = sim_domain.background_domain

    if_coords_list = []
    err_az_list = []
    err_bn_list = []
    err_ht_list = []

    if interface_evaluator is not None:
        for key, d in domains_dict.items():
            domain = d["domain_object"]
            net_key = resolve_domain_net_key(domain, model)
            outer_pts = d.get("outer_surface_points", d.get("surface_points"))
            if outer_pts is not None and outer_pts.shape[0] > 0 and net_key != 'background':
                pts_req = outer_pts.to(computation_device).detach().clone().requires_grad_(True)
                e_az, e_bn, e_ht = interface_evaluator.compute_pointwise_pair_components(
                    key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=pts_req
                )
                if_coords_list.append(pts_req.detach().cpu().numpy())
                err_az_list.append(e_az.detach().cpu().numpy().ravel())
                err_bn_list.append(e_bn.detach().cpu().numpy().ravel())
                err_ht_list.append(e_ht.detach().cpu().numpy().ravel())

        for pair_key, pair_data in shared_dict.items():
            s_pts = pair_data.get("interface_points")
            if s_pts is not None and s_pts.shape[0] > 0:
                pts_req = s_pts.to(computation_device).detach().clone().requires_grad_(True)
                e_az, e_bn, e_ht = interface_evaluator.compute_pointwise_pair_components(
                    key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                    key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts_req
                )
                if_coords_list.append(pts_req.detach().cpu().numpy())
                err_az_list.append(e_az.detach().cpu().numpy().ravel())
                err_bn_list.append(e_bn.detach().cpu().numpy().ravel())
                err_ht_list.append(e_ht.detach().cpu().numpy().ravel())

    if if_coords_list:
        if_coords_all = np.vstack(if_coords_list)
        err_az_all = np.concatenate(err_az_list)
        err_bn_all = np.concatenate(err_bn_list)
        err_ht_all = np.concatenate(err_ht_list)
    else:
        if_coords_all = np.empty((0, 2))
        err_az_all = np.empty((0,))
        err_bn_all = np.empty((0,))
        err_ht_all = np.empty((0,))

    fig, axs = plt.subplots(2, 5, figsize=(28, 10))
    n_levels = getattr(vis_cfg, 'contour_levels', 60)

    ax_pts = axs[0, 0]
    ax_pts.set_title("Categorized Collocation Points")
    
    domain_colors = ['#1a365d', '#742a2a', '#1c4532', '#4a5568', '#44337a', '#7b341e']
    pair_colors = ['#800000', '#004080', '#004d40', '#4a148c', '#bf360c']

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

    ax_pts.set_xlim(visualizer_instance.x_bounds)
    ax_pts.set_ylim(visualizer_instance.y_bounds)
    ax_pts.set_aspect('equal')
    
    res_min = vis_cfg.res_vmin if vis_cfg.res_vmin is not None else 1e-4
    res_max = vis_cfg.res_vmax if vis_cfg.res_vmax is not None else 10.0
    res_abs = np.clip(np.abs(residual_numpy), res_min, res_max)

    disc_levels = getattr(vis_cfg, 'res_discrete_levels', [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])

    if getattr(vis_cfg, 'res_scale', 'log') == 'log' and disc_levels is not None:
        base_cmap = plt.cm.get_cmap(vis_cfg.res_cmap, len(disc_levels) - 1)
        norm_res = mcolors.BoundaryNorm(disc_levels, ncolors=base_cmap.N, clip=True)
        levels_res = disc_levels
    else:
        base_cmap = vis_cfg.res_cmap
        levels_res = np.linspace(res_min, res_max, n_levels)
        norm_res = Normalize(vmin=res_min, vmax=res_max)

    contour_res = axs[0, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, res_abs,
        levels=levels_res, cmap=base_cmap, norm=norm_res
    )
    cbar_res = fig.colorbar(contour_res, ax=axs[0, 1], ticks=disc_levels if getattr(vis_cfg, 'res_scale', 'log') == 'log' else None)
    if getattr(vis_cfg, 'res_scale', 'log') == 'log':
        cbar_res.ax.set_yscale('log')
    cbar_res.set_label("Residual Magnitude", fontsize=9)
    axs[0, 1].set_title("PDE Residual Error (Discrete Log)")
    axs[0, 1].set_aspect('equal')

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
    
    axs[0, 3].plot(visualizer_instance.epoch_history, visualizer_instance.loss_history, 'b-')
    axs[0, 3].set_yscale('log')
    axs[0, 3].set_title("Loss Optimization")
    axs[0, 3].set_xlabel("Total Optimization Step")
    axs[0, 3].set_ylabel("Loss")
    axs[0, 3].grid(True, which="both", ls="-", alpha=0.2)

    az_min = vis_cfg.az_vmin if vis_cfg.az_vmin is not None else float(np.min(A_z_numpy))
    az_max = vis_cfg.az_vmax if vis_cfg.az_vmax is not None else float(np.max(A_z_numpy))
    levels_az = np.linspace(az_min, az_max, n_levels) if az_min != az_max else n_levels

    contour_az = axs[1, 0].contourf(visualizer_instance.X_grid, visualizer_instance.Y_grid, A_z_numpy, levels=levels_az, cmap=vis_cfg.az_cmap)
    fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
    axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
    axs[1, 0].set_aspect('equal')

    b_unit = getattr(vis_cfg, 'b_mag_unit', 'Tesla')
    b_scale = 1000.0 if b_unit == "mTesla" else 1.0
    B_mag_plot = B_mag_numpy * b_scale
    unit_str = "mTesla" if b_unit == "mTesla" else "T"

    b_min = vis_cfg.b_mag_vmin if vis_cfg.b_mag_vmin is not None else 0.0
    b_max = vis_cfg.b_mag_vmax if vis_cfg.b_mag_vmax is not None else (1800.0 if b_unit == "mTesla" else 1.8)
    b_mag_clamped = np.clip(B_mag_plot, b_min, b_max)
    num_div = getattr(vis_cfg, 'b_mag_number_of_division', None)

    if getattr(vis_cfg, 'b_mag_cmap', 'jet') == "ansys_fem":
        if num_div is not None and num_div > 0:
            cmap_b = mcolors.LinearSegmentedColormap.from_list("ansys_fem_interp", ANSYS_FEM_HEX_COLORS, N=num_div)
            levels_b = np.linspace(b_min, b_max, num_div + 1)
            norm_b = mcolors.BoundaryNorm(levels_b, ncolors=cmap_b.N, clip=True)
        else:
            cmap_b = mcolors.ListedColormap(ANSYS_FEM_HEX_COLORS)
            levels_b = np.linspace(b_min, b_max, 21)
            norm_b = mcolors.BoundaryNorm(levels_b, ncolors=cmap_b.N, clip=True)
    else:
        cmap_b = vis_cfg.b_mag_cmap
        levels_b = np.linspace(b_min, b_max, n_levels)
        norm_b = Normalize(vmin=b_min, vmax=b_max)

    contour_b = axs[1, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=cmap_b, norm=norm_b
    )
    fig.colorbar(contour_b, ax=axs[1, 1], label=f"|B| ({unit_str})")
    axs[1, 1].set_title(f"Flux Density Magnitude ($|B|$ [{b_min}-{b_max}{unit_str}])")
    axs[1, 1].set_aspect('equal')

    contour_b_bg = axs[1, 2].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=cmap_b, norm=norm_b, alpha=0.3
    )
    fig.colorbar(contour_b_bg, ax=axs[1, 2], label=f"|B| ({unit_str})")
    step = max(1, visualizer_instance.resolution // 20)
    B_sub = np.maximum(B_mag_numpy[::step, ::step], 1e-12)
    axs[1, 2].quiver(
        visualizer_instance.X_grid[::step, ::step], visualizer_instance.Y_grid[::step, ::step], 
        B_x_numpy[::step, ::step] / B_sub, B_y_numpy[::step, ::step] / B_sub, 
        scale=30.0, color='black', pivot='mid'
    )
    axs[1, 2].set_title("Flux Density Vectors (B)")
    axs[1, 2].set_aspect('equal')

    def plot_interface_loss_map(ax_target, err_vals, title_str, cbar_label):
        for dom in sim_domain.domains_list:
            if getattr(dom, 'domain_name', '') != 'background' and dom.geometry_config.outline is not None:
                outline = np.array(dom.geometry_config.outline)
                closed_poly = np.vstack([outline, outline[0]])
                ax_target.plot(closed_poly[:, 0], closed_poly[:, 1], color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)
        if if_coords_all.shape[0] > 0 and err_vals.shape[0] > 0:
            vmin_if = 1e-8
            vmax_if = max(float(np.max(err_vals)), 1e-4)
            err_clamped = np.clip(err_vals, vmin_if, vmax_if)
            sc = ax_target.scatter(
                if_coords_all[:, 0], if_coords_all[:, 1],
                c=err_clamped, cmap='jet', norm=LogNorm(vmin=vmin_if, vmax=vmax_if),
                s=6.0, marker='o', linewidths=0
            )
            fig.colorbar(sc, ax=ax_target, label=cbar_label)
        ax_target.set_xlim(visualizer_instance.x_bounds)
        ax_target.set_ylim(visualizer_instance.y_bounds)
        ax_target.set_title(title_str)
        ax_target.set_aspect('equal')

    plot_interface_loss_map(axs[0, 4], err_az_all, r"Interface Loss Map ($A_z$)", r"$(A_{z,a}^* - A_{z,b}^*)^2$")
    plot_interface_loss_map(axs[1, 3], err_bn_all, r"Interface Loss Map ($B_n$)", r"$w_c (B_{n,a}^* - B_{n,b}^*)^2$")
    plot_interface_loss_map(axs[1, 4], err_ht_all, r"Interface Loss Map ($H_t$)", r"$w_c (\Delta H_t^* / \nu)^2$")
    
    fig.suptitle(f"XPINN Training Process - Step {epoch}", fontsize=16)
    plt.tight_layout()
    
    frame_name = os.path.join(visualizer_instance.output_directory, f"frame_{len(visualizer_instance.frame_paths) + 1:05d}.png")
    plt.savefig(frame_name, dpi=100)
    plt.close(fig)
    
    visualizer_instance.frame_paths.append(frame_name)
    model.train()

    if getattr(vis_cfg, 'display_colab_frames', False):
        try:
            from IPython.display import Image, display, clear_output
            clear_output(wait=True)
            print(f"[*] Live Colab Frame at Step {epoch}: {frame_name}")
            display(Image(filename=frame_name))
        except Exception:
            pass