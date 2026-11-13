import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RectBivariateSpline

# 20 mã màu Hex chuẩn Ansys Maxwell FEM kế thừa từ mã nguồn PINN
ANSYS_FEM_HEX_COLORS = [
    '#0000FF', '#0036FF', '#006BFF', '#00A1FF', '#00D7FF', 
    '#00FFF2', '#00FFBC', '#00FF86', '#00FF51', '#00FF1B', 
    '#1BFF00', '#51FF00', '#86FF00', '#BCFF00', '#F2FF00', 
    '#FFD700', '#FFA100', '#FF6B00', '#FF3600', '#FF0000'
]

def plot_flux_density(
    toy_model_obj, 
    fld_path=None,
    output_png_path=None, 
    save_path=None,
    save_image_path=None,
    title="B-field Density Map (T)", 
    target_resolution=1500,
    sigma_smooth=0.0,
    smooth="cubic",
    transpose_axes=False,
    show=True,
    figsize=(6.4, 4.8),
    dpi=100,
    vmin=0.0,
    vmax=2.0,
    cmap="ansys_fem",
    num_divisions=100,
    **kwargs
):
    """
    Vẽ và hiển thị heatmap mật độ từ thông B kế thừa 20 mã màu Ansys FEM chuẩn từ PINN.
    """
    record = getattr(toy_model_obj, "record", None)
    field_record = getattr(record, "flux_density_field", None) if record else None

    if field_record is None:
        if fld_path is None and hasattr(toy_model_obj, "configuration"):
            fld_path = getattr(toy_model_obj.configuration.analysis, "b_fld_path", None)
            
        if fld_path and os.path.exists(fld_path):
            from .export_flux_density import parse_fld_file
            field_record = parse_fld_file(fld_path)
            if record:
                record.flux_density_field = field_record

    if field_record is None:
        raise AttributeError("Không tìm thấy dữ liệu flux_density_field trong toy_model.record và không tìm thấy file .fld hợp lệ.")

    unique_x = field_record.x
    unique_y = field_record.y
    grid_b = field_record.b_mag

    if transpose_axes:
        grid_b = grid_b.T
        unique_x, unique_y = unique_y, unique_x

    if len(unique_x) < 2 or len(unique_y) < 2:
        raise ValueError(f"Dữ liệu lưới không đủ để vẽ heatmap: nx={len(unique_x)}, ny={len(unique_y)}")

    res = target_resolution if (target_resolution and target_resolution > 0) else 1000
    x_dense = np.linspace(unique_x.min(), unique_x.max(), res)
    y_dense = np.linspace(unique_y.min(), unique_y.max(), res)

    try:
        spline_func = RectBivariateSpline(unique_y, unique_x, grid_b, kx=3, ky=3)
        grid_b_dense = spline_func(y_dense, x_dense)
    except Exception:
        grid_b_dense = grid_b

    if sigma_smooth and sigma_smooth > 0:
        from scipy.ndimage import gaussian_filter
        grid_b_dense = gaussian_filter(grid_b_dense, sigma=sigma_smooth)

    # Cấu hình Bảng màu Ansys FEM kế thừa từ mã nguồn PINN
    if cmap == "ansys_fem":
        if num_divisions is not None and num_divisions > 0:
            cmap_obj = mcolors.LinearSegmentedColormap.from_list("ansys_fem_interp", ANSYS_FEM_HEX_COLORS, N=num_divisions)
            levels = np.linspace(vmin, vmax, num_divisions + 1)
            norm_obj = mcolors.BoundaryNorm(levels, ncolors=cmap_obj.N, clip=True)
        else:
            cmap_obj = mcolors.ListedColormap(ANSYS_FEM_HEX_COLORS)
            levels = np.linspace(vmin, vmax, 21)
            norm_obj = mcolors.BoundaryNorm(levels, ncolors=cmap_obj.N, clip=True)
    else:
        cmap_obj = cmap
        norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    extent = [unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()]
    im = ax.imshow(
        grid_b_dense, 
        extent=extent, 
        origin='lower', 
        cmap=cmap_obj, 
        norm=norm_obj,
        interpolation='bicubic',
        aspect='equal'
    )

    cbar = fig.colorbar(im, ax=ax, label='|B| (Tesla)', pad=0.03)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('X (mm)', fontsize=9)
    ax.set_ylabel('Y (mm)', fontsize=9)

    ax.grid(False)
    plt.tight_layout()
    
    if show:
        print(f"🖼️ Đang hiển thị heatmap mật độ từ thông Ansys FEM ({figsize[0]}x{figsize[1]} in, scale {vmin}-{vmax} T)...")
        plt.show()

    return fig