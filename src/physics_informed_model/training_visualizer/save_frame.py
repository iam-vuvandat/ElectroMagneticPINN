import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY
from training_visualizer.resolve_domain_net_key import resolve_domain_net_key
from training_engine.compute_loss import _compute_outer_box_normals

ANSYS_FEM_HEX_COLORS = [
    '#0000FF', '#0036FF', '#006BFF', '#00A1FF', '#00D7FF',
    '#00FFF2', '#00FFBC', '#00FF86', '#00FF51', '#00FF1B',
    '#1BFF00', '#51FF00', '#86FF00', '#BCFF00', '#F2FF00',
    '#FFD700', '#FFA100', '#FF6B00', '#FF3600', '#FF0000'
]


def save_frame(visualizer_instance, epoch, loss_value, domains_data, loss_breakdown=None, fpi_step=None):
    tm = getattr(visualizer_instance.electro_magnetic_pinn, 'training_manager_instance', None)
    if loss_breakdown is None and tm is not None:
        loss_breakdown = getattr(tm, 'last_loss_breakdown', {})
    if loss_breakdown is None:
        loss_breakdown = {}
    if fpi_step is None:
        fpi_step = getattr(tm, 'current_fpi_step', 0) if tm is not None else 0

    visualizer_instance.loss_history.append(loss_value)
    visualizer_instance.epoch_history.append(epoch)

    pde_val = max(float(loss_breakdown.get('pde_loss', loss_value)), 1e-16)
    const_val = max(float(loss_breakdown.get('const_loss', 1e-16)), 1e-16)
    bn_val = max(float(loss_breakdown.get('interface_Bn', 1e-16)), 1e-16)
    ht_val = max(float(loss_breakdown.get('interface_Ht', 1e-16)), 1e-16)
    bc_val = max(float(loss_breakdown.get('boundary_loss', 1e-16)), 1e-16)

    visualizer_instance.pde_loss_history.append(pde_val)
    visualizer_instance.const_loss_history.append(const_val)
    visualizer_instance.bn_loss_history.append(bn_val)
    visualizer_instance.ht_loss_history.append(ht_val)
    visualizer_instance.bc_loss_history.append(bc_val)

    lmb = loss_breakdown.get('relobralo_lambdas', {})
    visualizer_instance.w_bn_history.append(float(lmb.get('loss_bn', 1.0)))
    visualizer_instance.w_ht_history.append(float(lmb.get('loss_ht', 1.0)))
    visualizer_instance.w_bc_history.append(float(lmb.get('loss_bc', 1.0)))

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

    B_x_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    B_y_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    H_x_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    H_y_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    nu_phys_all = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    residual_total = torch.zeros((eval_points.shape[0], 1), device=computation_device)
    const_residual_total = torch.zeros((eval_points.shape[0], 1), device=computation_device)

    for domain in sim_domain.domains_list:
        d_id = domain.domain_id
        net_key = resolve_domain_net_key(domain, model)
        mask_bool = (mat_id == float(d_id))
        if mask_bool.any():
            field_star = model.forward_network(net_key, eval_points)
            Bx_sub_phys = field_star[:, 0:1] * sim_domain.B0
            By_sub_phys = field_star[:, 1:2] * sim_domain.B0

            nu_star_net = None
            if getattr(domain.physics_config, 'is_nonlinear', False):
                if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                    nu_star_pred = model.forward_nu_network(net_key, eval_points)
                    nu_star_net = nu_star_pred
                    nu_dom = nu_star_pred * sim_domain.nu0
                else:
                    B_mag_sub = torch.sqrt(Bx_sub_phys**2 + By_sub_phys**2 + 1e-12)
                    nu_dom = domain.evaluate_reluctivity(eval_points, B_magnitude_tensor=B_mag_sub, is_nonlinear=True).to(computation_device)
            else:
                nu_dom = domain.evaluate_reluctivity(eval_points, is_nonlinear=False).to(computation_device)

            hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
            hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
            Hx_sub_phys = nu_dom * Bx_sub_phys - hx_dom
            Hy_sub_phys = nu_dom * By_sub_phys - hy_dom
            jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

            res_div, res_curl, res_const_x, res_const_y = pde_evaluator.compute_first_order_pde_residuals(
                eval_points, field_star, nu_dom, jz_dom, hx_dom, hy_dom, nu_star_net=nu_star_net
            )
            res_mag = torch.sqrt((res_div ** 2) + (res_curl ** 2) + (res_const_x ** 2) + (res_const_y ** 2))
            const_mag = torch.sqrt((res_const_x ** 2) + (res_const_y ** 2))

            B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
            B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
            H_x_phys = torch.where(mask_bool, Hx_sub_phys.detach(), H_x_phys)
            H_y_phys = torch.where(mask_bool, Hy_sub_phys.detach(), H_y_phys)
            nu_phys_all = torch.where(mask_bool, nu_dom.detach(), nu_phys_all)
            residual_total = torch.where(mask_bool, res_mag.detach(), residual_total)
            const_residual_total = torch.where(mask_bool, const_mag.detach(), const_residual_total)

    B_x_numpy = B_x_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_y_numpy = B_y_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)

    H_x_numpy = H_x_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    H_y_numpy = H_y_phys.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    H_mag_numpy = np.sqrt(H_x_numpy ** 2 + H_y_numpy ** 2)

    mu_r_numpy = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys_all, min=1e-6)).cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    residual_numpy = residual_total.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)
    const_residual_numpy = const_residual_total.cpu().numpy().reshape(visualizer_instance.resolution, visualizer_instance.resolution)

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    bg_domain = sim_domain.background_domain

    if_coords_list = []
    err_bn_list = []
    err_ht_list = []

    if interface_evaluator is not None:
        with torch.no_grad():
            for key, d in domains_dict.items():
                domain = d["domain_object"]
                net_key = resolve_domain_net_key(domain, model)
                outer_pts = d.get("outer_surface_points", d.get("surface_points"))
                if outer_pts is not None and outer_pts.shape[0] > 0 and net_key != 'background':
                    pts_req = outer_pts.to(computation_device)
                    e_bn, e_ht = interface_evaluator.compute_pointwise_pair_components(
                        key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=pts_req
                    )
                    if_coords_list.append(pts_req.cpu().numpy())
                    err_bn_list.append(e_bn.cpu().numpy().ravel())
                    err_ht_list.append(e_ht.cpu().numpy().ravel())

            for pair_key, pair_data in shared_dict.items():
                s_pts = pair_data.get("interface_points")
                if s_pts is not None and s_pts.shape[0] > 0:
                    pts_req = s_pts.to(computation_device)
                    e_bn, e_ht = interface_evaluator.compute_pointwise_pair_components(
                        key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                        key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts_req
                    )
                    if_coords_list.append(pts_req.cpu().numpy())
                    err_bn_list.append(e_bn.cpu().numpy().ravel())
                    err_ht_list.append(e_ht.cpu().numpy().ravel())

    if if_coords_list:
        if_coords_all = np.vstack(if_coords_list)
        err_bn_all = np.concatenate(err_bn_list)
        err_ht_all = np.concatenate(err_ht_list)
    else:
        if_coords_all = np.empty((0, 2))
        err_bn_all = np.empty((0,))
        err_ht_all = np.empty((0,))

    bc_pts = domains_data.get("boundary_points")
    bc_coords_np = np.empty((0, 2))
    err_bc_all = np.empty((0,))
    boundary_obj = getattr(sim_domain, 'boundary', None)
    enforce_method = getattr(boundary_obj.physics_config, 'enforce_method', 'hard_factor') if boundary_obj else 'hard_factor'

    if bc_pts is not None and bc_pts.shape[0] > 0:
        bc_pts_dev = bc_pts.to(computation_device)
        bc_coords_np = bc_pts.cpu().detach().numpy()
        nx_bc, ny_bc = _compute_outer_box_normals(
            bc_pts_dev, sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple
        )
        bc_phys_props = sim_domain.evaluate_global_physical_properties(bc_pts_dev)
        bc_mat_id = bc_phys_props["material_classification"].view(-1)

        err_bc_list = []
        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = resolve_domain_net_key(domain, model)
            mask = (bc_mat_id == float(d_id))
            if mask.any():
                sub_pts = bc_pts_dev[mask]
                with torch.no_grad():
                    field_bc_pred = model.forward_network(net_key, sub_pts)
                    bn_bc_pred = field_bc_pred[:, 0:1] * nx_bc[mask] + field_bc_pred[:, 1:2] * ny_bc[mask]
                    sq_err = bn_bc_pred ** 2
                err_bc_list.append(sq_err.cpu().view(-1).numpy())
        if err_bc_list:
            err_bc_all = np.concatenate(err_bc_list)

    fig, axs = plt.subplots(3, 4, figsize=(24, 15))
    n_levels = getattr(vis_cfg, 'contour_levels', 60)

    def draw_domain_outlines(ax_target, color='#111111', linestyle='-', linewidth=1.0, alpha=0.85):
        for dom in sim_domain.domains_list:
            if getattr(dom, 'domain_name', '') != 'background' and dom.geometry_config.outline is not None:
                outline = np.array(dom.geometry_config.outline)
                closed_poly = np.vstack([outline, outline[0]])
                ax_target.plot(
                    closed_poly[:, 0], closed_poly[:, 1],
                    color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=5
                )

    ax_pts = axs[0, 0]
    ax_pts.set_title("Collocation point", fontsize=11, fontweight='bold')
    domain_colors = ['#1a365d', '#742a2a', '#1c4532', '#4a5568', '#44337a', '#7b341e']
    pair_colors = ['#800000', '#004080', '#004d40', '#4a148c', '#bf360c']

    for idx, (key, d) in enumerate(domains_dict.items()):
        c_dom = domain_colors[idx % len(domain_colors)]
        int_pts = d.get("interior_points")
        if int_pts is not None and int_pts.shape[0] > 0:
            pts_np = int_pts.detach().cpu().numpy()
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=0.6, color=c_dom, marker='o', linewidths=0, alpha=0.55)
        surf_pts = d.get("outer_surface_points", d.get("surface_points"))
        if surf_pts is not None and surf_pts.shape[0] > 0:
            pts_np = surf_pts.detach().cpu().numpy()
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=1.2, color=c_dom, marker='o', linewidths=0, alpha=0.85)

    for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
        if_pts = pair_data.get("interface_points")
        if if_pts is not None and if_pts.shape[0] > 0:
            pts_np = if_pts.detach().cpu().numpy()
            c_pair = pair_colors[p_idx % len(pair_colors)]
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=1.2, color=c_pair, marker='o', linewidths=0, alpha=0.95)

    if bc_pts is not None and bc_pts.shape[0] > 0:
        ax_pts.scatter(bc_coords_np[:, 0], bc_coords_np[:, 1], s=1.5, color='#000000', marker='o', linewidths=0, alpha=0.9)

    draw_domain_outlines(ax_pts, color='#555555', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_pts.set_xlim(visualizer_instance.x_bounds)
    ax_pts.set_ylim(visualizer_instance.y_bounds)
    ax_pts.set_aspect('equal')

    disc_levels = getattr(vis_cfg, 'res_discrete_levels', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    res_min = vis_cfg.res_vmin if vis_cfg.res_vmin is not None else float(disc_levels[0])
    res_max = vis_cfg.res_vmax if vis_cfg.res_vmax is not None else float(disc_levels[-1])
    res_scale_mode = getattr(vis_cfg, 'res_scale', 'log')

    if res_scale_mode == 'log' and disc_levels is not None:
        base_cmap = plt.get_cmap(vis_cfg.res_cmap, len(disc_levels) - 1)
        norm_res = mcolors.BoundaryNorm(disc_levels, ncolors=base_cmap.N, clip=True)
        levels_res = disc_levels
    elif res_scale_mode == 'log':
        base_cmap = plt.get_cmap(vis_cfg.res_cmap)
        norm_res = LogNorm(vmin=res_min, vmax=res_max, clip=True)
        levels_res = np.logspace(np.log10(res_min), np.log10(res_max), n_levels)
    else:
        base_cmap = plt.get_cmap(vis_cfg.res_cmap)
        levels_res = np.linspace(res_min, res_max, n_levels)
        norm_res = Normalize(vmin=res_min, vmax=res_max, clip=True)

    res_abs = np.clip(np.abs(residual_numpy), res_min, res_max)
    contour_res = axs[0, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, res_abs,
        levels=levels_res, cmap=base_cmap, norm=norm_res
    )
    cbar_res = fig.colorbar(
        contour_res,
        ax=axs[0, 1],
        ticks=disc_levels if (res_scale_mode == 'log' and disc_levels is not None) else None
    )
    if res_scale_mode == 'log':
        cbar_res.ax.set_yscale('log')
    cbar_res.set_label("Residual Magnitude", fontsize=9)
    draw_domain_outlines(axs[0, 1], color='#FFFFFF', linestyle='--', linewidth=0.7, alpha=0.5)
    axs[0, 1].set_title("PDE residual map", fontsize=11, fontweight='bold')
    axs[0, 1].set_aspect('equal')

    has_bc_plot = (enforce_method == 'soft_loss') or ('loss_bc' in lmb)
    ax_loss = axs[0, 2]
    steps = visualizer_instance.epoch_history
    ax_loss.plot(steps, visualizer_instance.pde_loss_history, label="PDE loss", linewidth=1.4)
    ax_loss.plot(steps, visualizer_instance.const_loss_history, label="Constitutive loss", linewidth=1.4)
    ax_loss.plot(steps, visualizer_instance.bn_loss_history, label=r"$B_n$ loss", linewidth=1.4)
    ax_loss.plot(steps, visualizer_instance.ht_loss_history, label=r"$H_t$ loss", linewidth=1.4)
    if has_bc_plot:
        ax_loss.plot(steps, visualizer_instance.bc_loss_history, label="BC loss", linewidth=1.4)
    ax_loss.set_yscale('log')
    ax_loss.set_title("Loss optimization", fontsize=11, fontweight='bold')
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, which="both", ls="--", alpha=0.25)
    ax_loss.legend(loc="upper right", fontsize=8.5)

    ax_w = axs[0, 3]
    ax_w.plot(steps, visualizer_instance.w_bn_history, label=r"$B_n$", linewidth=1.4)
    ax_w.plot(steps, visualizer_instance.w_ht_history, label=r"$H_t$", linewidth=1.4)
    if has_bc_plot:
        ax_w.plot(steps, visualizer_instance.w_bc_history, label="BC", linewidth=1.4)
    ax_w.set_title("Loss weight", fontsize=11, fontweight='bold')
    ax_w.set_xlabel("Step")
    ax_w.set_ylabel("Weight")
    ax_w.grid(True, which="both", ls="--", alpha=0.25)
    ax_w.legend(loc="upper right", fontsize=8.5)

    if_levels = getattr(vis_cfg, 'if_discrete_levels', [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    if_min = getattr(vis_cfg, 'if_vmin', None)
    if_min = float(if_min) if if_min is not None else float(if_levels[0])
    if_max = getattr(vis_cfg, 'if_vmax', None)
    if_max = float(if_max) if if_max is not None else float(if_levels[-1])

    if res_scale_mode == 'log' and if_levels is not None:
        if_cmap = plt.get_cmap(vis_cfg.res_cmap, len(if_levels) - 1)
        norm_if = mcolors.BoundaryNorm(if_levels, ncolors=if_cmap.N, clip=True)
    elif res_scale_mode == 'log':
        if_cmap = plt.get_cmap(vis_cfg.res_cmap)
        norm_if = LogNorm(vmin=if_min, vmax=if_max, clip=True)
    else:
        if_cmap = plt.get_cmap(vis_cfg.res_cmap)
        norm_if = Normalize(vmin=if_min, vmax=if_max, clip=True)

    def plot_scatter_loss_map(ax_target, coords, err_vals, title_str, cbar_label):
        draw_domain_outlines(ax_target, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)
        if coords.shape[0] > 0 and err_vals.shape[0] > 0:
            err_clamped = np.clip(err_vals, if_min, if_max)
            sc = ax_target.scatter(
                coords[:, 0], coords[:, 1],
                c=err_clamped, cmap=if_cmap, norm=norm_if,
                s=6.0, marker='o', linewidths=0
            )
            cbar_if = fig.colorbar(
                sc,
                ax=ax_target,
                ticks=if_levels if (res_scale_mode == 'log' and if_levels is not None) else None
            )
            if res_scale_mode == 'log':
                cbar_if.ax.set_yscale('log')
            cbar_if.set_label(cbar_label, fontsize=9)
        ax_target.set_xlim(visualizer_instance.x_bounds)
        ax_target.set_ylim(visualizer_instance.y_bounds)
        ax_target.set_title(title_str, fontsize=11, fontweight='bold')
        ax_target.set_aspect('equal')

    plot_scatter_loss_map(axs[1, 0], bc_coords_np, err_bc_all, r"Boundary loss map", r"$(B_{n}^*)^2$")

    const_abs = np.clip(np.abs(const_residual_numpy), res_min, res_max)
    contour_const = axs[1, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, const_abs,
        levels=levels_res, cmap=base_cmap, norm=norm_res
    )
    cbar_const = fig.colorbar(
        contour_const,
        ax=axs[1, 1],
        ticks=disc_levels if (res_scale_mode == 'log' and disc_levels is not None) else None
    )
    if res_scale_mode == 'log':
        cbar_const.ax.set_yscale('log')
    cbar_const.set_label("Constitutive Residual", fontsize=9)
    draw_domain_outlines(axs[1, 1], color='#FFFFFF', linestyle='--', linewidth=0.7, alpha=0.5)
    axs[1, 1].set_title("Constitutive residual map", fontsize=11, fontweight='bold')
    axs[1, 1].set_aspect('equal')

    plot_scatter_loss_map(axs[1, 2], if_coords_all, err_bn_all, r"$B_n$ interface loss map", r"$w_c (B_{n,a}^* - B_{n,b}^*)^2$")

    plot_scatter_loss_map(axs[1, 3], if_coords_all, err_ht_all, r"$H_t$ interface loss map", r"$w_c (\Delta H_t^* / \nu)^2$")

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

    contour_mur = axs[2, 0].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, mu_r_clamped, levels=levels_mur,
        cmap=vis_cfg.mur_cmap, norm=norm_mur
    )
    fig.colorbar(contour_mur, ax=axs[2, 0], label=lbl_mur)
    draw_domain_outlines(axs[2, 0], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 0].set_title("Permeability map", fontsize=11, fontweight='bold')
    axs[2, 0].set_aspect('equal')

    h_min = float(np.min(H_mag_numpy))
    h_max = float(np.max(H_mag_numpy))
    levels_h = np.linspace(h_min, h_max, n_levels) if h_min != h_max else n_levels

    contour_h = axs[2, 1].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, H_mag_numpy, levels=levels_h, cmap=vis_cfg.az_cmap
    )
    fig.colorbar(contour_h, ax=axs[2, 1], label=r"$|H|$ (A/m)")
    draw_domain_outlines(axs[2, 1], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 1].set_title(r"|H| map", fontsize=11, fontweight='bold')
    axs[2, 1].set_aspect('equal')

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

    contour_b = axs[2, 2].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=cmap_b, norm=norm_b
    )
    fig.colorbar(contour_b, ax=axs[2, 2], label=f"|B| ({unit_str})")
    draw_domain_outlines(axs[2, 2], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 2].set_title(r"|B| map", fontsize=11, fontweight='bold')
    axs[2, 2].set_aspect('equal')

    contour_b_bg = axs[2, 3].contourf(
        visualizer_instance.X_grid, visualizer_instance.Y_grid, b_mag_clamped, levels=levels_b,
        cmap=cmap_b, norm=norm_b, alpha=0.3
    )
    fig.colorbar(contour_b_bg, ax=axs[2, 3], label=f"|B| ({unit_str})")
    step = max(1, visualizer_instance.resolution // 20)
    B_sub = np.maximum(B_mag_numpy[::step, ::step], 1e-12)
    axs[2, 3].quiver(
        visualizer_instance.X_grid[::step, ::step], visualizer_instance.Y_grid[::step, ::step],
        B_x_numpy[::step, ::step] / B_sub, B_y_numpy[::step, ::step] / B_sub,
        scale=30.0, color='black', pivot='mid'
    )
    draw_domain_outlines(axs[2, 3], color='#111111', linestyle='--', linewidth=0.7, alpha=0.6)
    axs[2, 3].set_title("B vector map", fontsize=11, fontweight='bold')
    axs[2, 3].set_aspect('equal')

    fig.suptitle(f"Training process: Step {epoch}; FPI step {fpi_step}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    frame_name = os.path.join(visualizer_instance.output_directory, f"frame_{len(visualizer_instance.frame_paths) + 1:05d}.png")
    plt.savefig(frame_name, dpi=100)
    plt.close(fig)

    visualizer_instance.frame_paths.append(frame_name)
    model.train()

    if getattr(vis_cfg, 'display_colab_frames', False):
        try:
            from IPython.display import Image, display, clear_output
            clear_output(wait=True)
            print(f"[*] Live Colab Frame at Step {epoch} (FPI step {fpi_step}): {frame_name}")
            display(Image(filename=frame_name))
        except Exception:
            pass