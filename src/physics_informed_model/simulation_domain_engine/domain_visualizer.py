import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

def display_simulation_domain(simulation_domain_instance, x_boundaries_tuple=None, y_boundaries_tuple=None, resolution=150):
    if x_boundaries_tuple is None:
        x_boundaries_tuple = simulation_domain_instance.x_boundaries_tuple
    if y_boundaries_tuple is None:
        y_boundaries_tuple = simulation_domain_instance.y_boundaries_tuple

    x_coords = np.linspace(x_boundaries_tuple[0], x_boundaries_tuple[1], resolution)
    y_coords = np.linspace(y_boundaries_tuple[0], y_boundaries_tuple[1], resolution)
    dx = (x_boundaries_tuple[1] - x_boundaries_tuple[0]) / (resolution - 1)
    dy = (y_boundaries_tuple[1] - y_boundaries_tuple[0]) / (resolution - 1)
    
    # Mở rộng extent nửa bước lưới để tâm pixel khớp hoàn toàn với đường biên hình học
    exact_extent = [
        x_boundaries_tuple[0] - dx / 2.0,
        x_boundaries_tuple[1] + dx / 2.0,
        y_boundaries_tuple[0] - dy / 2.0,
        y_boundaries_tuple[1] + dy / 2.0
    ]

    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    xy_points_tensor = torch.tensor(np.column_stack((X_grid.ravel(), Y_grid.ravel())), dtype=torch.float32)
    physical_properties = simulation_domain_instance.evaluate_global_physical_properties(xy_points_tensor)
    
    mat_id_tensor = physical_properties["material_classification"]
    mat_id_grid = mat_id_tensor.numpy().reshape(resolution, resolution)
    
    reluctivity_tensor = physical_properties["reluctivity"]
    mu_r_tensor = VACUUM_RELUCTIVITY / reluctivity_tensor
    mu_r_grid = mu_r_tensor.numpy().reshape(resolution, resolution)
    
    hx_tensor = physical_properties["coercive_field_x"]
    hy_tensor = physical_properties["coercive_field_y"]
    hx_grid = hx_tensor.numpy().reshape(resolution, resolution)
    hy_grid = hy_tensor.numpy().reshape(resolution, resolution)
    hc_mag_grid = np.sqrt(hx_grid**2 + hy_grid**2)
    
    jz_tensor = physical_properties["current_density_z"]
    jz_grid = jz_tensor.numpy().reshape(resolution, resolution)
    jz_mag_grid = np.abs(jz_grid)
    
    total_domains = simulation_domain_instance.get_domain_count()
    
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })
    
    fig, axs = plt.subplots(2, 2, figsize=(13, 11), facecolor='white')

    def draw_outlines(ax):
        for domain in simulation_domain_instance.domains_list:
            if getattr(domain, 'domain_name', '') != 'background' and domain.geometry_config.outline is not None:
                outline = np.array(domain.geometry_config.outline)
                closed_poly = np.vstack([outline, outline[0]])
                ax.plot(closed_poly[:, 0], closed_poly[:, 1], color='#111111', linestyle='--', linewidth=1.2, alpha=0.9, zorder=5)

    # 1. Material Layout
    palette = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1']
    color_list = [palette[(i - 1) % len(palette)] for i in range(1, total_domains)]
    color_list.append('#F2F4F7')
    
    cmap_discrete = ListedColormap(color_list)
    bounds_discrete = np.arange(0.5, total_domains + 1.5, 1.0)
    norm_discrete = BoundaryNorm(bounds_discrete, cmap_discrete.N)
    
    im_mat = axs[0, 0].imshow(mat_id_grid, extent=exact_extent, origin='lower', interpolation='nearest', cmap=cmap_discrete, norm=norm_discrete)
    cbar_mat = fig.colorbar(im_mat, ax=axs[0, 0], ticks=np.arange(1, total_domains + 1))
    cbar_mat.set_label("Material Domain Index")
    
    labels = [d.domain_name for d in simulation_domain_instance.domains_list]
    cbar_mat.set_ticklabels([f"{i+1}: {labels[i]}" for i in range(total_domains)])
    
    draw_outlines(axs[0, 0])
    axs[0, 0].set_xlim(x_boundaries_tuple)
    axs[0, 0].set_ylim(y_boundaries_tuple)
    axs[0, 0].set_title("Domain & Material Classification")
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[0, 0].set_aspect('equal')
    axs[0, 0].grid(True, linestyle=':', alpha=0.4)

    # 2. Relative Permeability
    im_mur = axs[0, 1].imshow(mu_r_grid, extent=exact_extent, origin='lower', interpolation='nearest', cmap="cividis")
    fig.colorbar(im_mur, ax=axs[0, 1], label=r"Relative Permeability ($\mu_r$)")
    draw_outlines(axs[0, 1])
    axs[0, 1].set_xlim(x_boundaries_tuple)
    axs[0, 1].set_ylim(y_boundaries_tuple)
    axs[0, 1].set_title(r"Permeability Distribution ($\mu_r$)")
    axs[0, 1].set_xlabel("x (m)")
    axs[0, 1].set_ylabel("y (m)")
    axs[0, 1].set_aspect('equal')
    axs[0, 1].grid(True, linestyle=':', alpha=0.4)

    # 3. Magnetization Vector Field (Mũi tên thanh mảnh, chuẩn tỷ lệ)
    step = max(1, resolution // 35)
    X_sub = X_grid[::step, ::step]
    Y_sub = Y_grid[::step, ::step]
    Hx_sub = hx_grid[::step, ::step]
    Hy_sub = hy_grid[::step, ::step]
    Hc_mag_sub = hc_mag_grid[::step, ::step]
    
    axs[1, 0].set_facecolor('#FAFAFA')
    draw_outlines(axs[1, 0])
    hc_mask = Hc_mag_sub > 1.0
    
    if np.any(hc_mask):
        norm_hc = mcolors.Normalize(vmin=np.min(Hc_mag_sub[hc_mask]) * 0.9, vmax=np.max(Hc_mag_sub[hc_mask]))
        cmap_hc = cm.turbo
        colors_hc = cmap_hc(norm_hc(Hc_mag_sub[hc_mask]))
        
        dir_x = Hx_sub[hc_mask] / Hc_mag_sub[hc_mask]
        dir_y = Hy_sub[hc_mask] / Hc_mag_sub[hc_mask]
        
        axs[1, 0].quiver(
            X_sub[hc_mask], Y_sub[hc_mask],
            dir_x, dir_y,
            color=colors_hc,
            pivot='mid',
            scale=45,            # Tăng scale để mũi tên ngắn lại vừa vặn
            width=0.0035,        # Giảm độ rộng thân mũi tên
            headwidth=3.2,       # Thu nhỏ bề ngang đầu mũi tên
            headlength=3.8,      # Thu gọn chiều dài đầu mũi tên
            headaxislength=3.2,
            zorder=4
        )
        sm_hc = cm.ScalarMappable(norm=norm_hc, cmap=cmap_hc)
        sm_hc.set_array([])
        fig.colorbar(sm_hc, ax=axs[1, 0], label=r"Coercive Field Magnitude $H_c$ (A/m)")
    else:
        axs[1, 0].text(0.5, 0.5, "No Permanent Magnetization ($H_c = 0$)", color='#666666', ha='center', va='center', transform=axs[1, 0].transAxes)
        
    axs[1, 0].set_xlim(x_boundaries_tuple)
    axs[1, 0].set_ylim(y_boundaries_tuple)
    axs[1, 0].set_title(r"Magnetization Vector Field ($\mathbf{H}_c$)")
    axs[1, 0].set_xlabel("x (m)")
    axs[1, 0].set_ylabel("y (m)")
    axs[1, 0].set_aspect('equal')
    axs[1, 0].grid(True, linestyle=':', alpha=0.4)

    # 4. Current Direction & Density
    axs[1, 1].set_facecolor('#FAFAFA')
    draw_outlines(axs[1, 1])
    Jz_sub = jz_grid[::step, ::step]
    Jz_mag_sub = jz_mag_grid[::step, ::step]
    jz_mask = Jz_mag_sub > 1.0
    
    if np.any(jz_mask):
        norm_jz = mcolors.Normalize(vmin=np.min(Jz_mag_sub[jz_mask]) * 0.9, vmax=np.max(Jz_mag_sub[jz_mask]))
        cmap_jz = cm.inferno
        colors_jz = cmap_jz(norm_jz(Jz_mag_sub[jz_mask]))
        
        dir_jz_x = np.zeros_like(Jz_sub[jz_mask])
        dir_jz_y = np.sign(Jz_sub[jz_mask])
        
        axs[1, 1].quiver(
            X_sub[jz_mask], Y_sub[jz_mask],
            dir_jz_x, dir_jz_y,
            color=colors_jz,
            pivot='mid',
            scale=45,
            width=0.0035,
            headwidth=3.2,
            headlength=3.8,
            headaxislength=3.2,
            zorder=4
        )
        sm_jz = cm.ScalarMappable(norm=norm_jz, cmap=cmap_jz)
        sm_jz.set_array([])
        fig.colorbar(sm_jz, ax=axs[1, 1], label=r"Current Density Magnitude $|J_z|$ (A/m$^2$)")
    else:
        axs[1, 1].text(0.5, 0.5, "No Current Density ($J_z = 0$)", color='#666666', ha='center', va='center', transform=axs[1, 1].transAxes)

    axs[1, 1].set_xlim(x_boundaries_tuple)
    axs[1, 1].set_ylim(y_boundaries_tuple)
    axs[1, 1].set_title(r"Conductor Current Distribution ($\mathbf{J}_z$)")
    axs[1, 1].set_xlabel("x (m)")
    axs[1, 1].set_ylabel("y (m)")
    axs[1, 1].set_aspect('equal')
    axs[1, 1].grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()