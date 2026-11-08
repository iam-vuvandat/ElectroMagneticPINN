import os
from pathlib import Path

# Xác định thư mục gốc dự án (Project Root)
current_script = Path(__file__).resolve()
project_root = (
    current_script.parent.parent
    if current_script.parent.name == "utils"
    else current_script.parent
)

toy_model_dir = (
    project_root / "src" / "ansys_aedt" / "maxwell_2d" / "toy_model_1"
)
pinn_dir = project_root / "src" / "physics_informed_model"
scenarios_dir = project_root / "scenarios"

print(
    "🚀 Đang khởi chạy cập nhật tự động và đồng loạt tại:"
    f" {project_root}\n"
)

# ==============================================================================
# 1. scenarios/excute1.py (Bộc lộ toàn bộ config, cài B = 0-2T, figsize chuẩn)
# ==============================================================================
EXCUTE1_CONTENT = """from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ansys_aedt.maxwell_2d.toy_model_1.toy_model_1 import ToyModel1

def main():
    toy_model = ToyModel1()
    config = toy_model.configuration

    # --- CẤU HÌNH THÔNG TIN CHUNG ---
    config.name = "toy_model_1"
    config.version = "2025.2"
    config.solution_type = "Magnetostatic"

    # --- CẤU HÌNH HÌNH HỌC (GEOMETRY) ---
    config.geometry.yoke.x_start = "-60mm"
    config.geometry.yoke.y_start = "-30mm"
    config.geometry.yoke.width = "120mm"
    config.geometry.yoke.height = "30mm"

    config.geometry.magnet_left.x_start = "-50mm"
    config.geometry.magnet_left.y_start = "0mm"
    config.geometry.magnet_left.width = "45mm"
    config.geometry.magnet_left.height = "18mm"
    config.geometry.magnet_left.coercive_direction = (0, 1)

    config.geometry.magnet_right.x_start = "5mm"
    config.geometry.magnet_right.y_start = "0mm"
    config.geometry.magnet_right.width = "45mm"
    config.geometry.magnet_right.height = "18mm"
    config.geometry.magnet_right.coercive_direction = (0, -1)

    config.geometry.region.x_start = "-120mm"
    config.geometry.region.y_start = "-120mm"
    config.geometry.region.width = "240mm"
    config.geometry.region.height = "240mm"

    # --- CẤU HÌNH VẬT LIỆU (MATERIALS) ---
    config.materials.yoke_material = "steel_1008"
    config.materials.magnet_material_left = "NdFe30 - Copy"
    config.materials.magnet_material_right = "NdFe30 - Copy (2)"
    config.materials.region_material = "vacuum"
    config.materials.hc_magnitude = 838000.0

    # --- CẤU HÌNH PHÂN TÍCH & TRÍCH XUẤT LƯỚI (ANALYSIS) ---
    config.analysis.setup_name = "Setup1"
    config.analysis.max_passes = 10
    config.analysis.min_passes = 2
    config.analysis.percent_error = 1.0
    config.analysis.max_element_length = "0.25mm"
    config.analysis.b_fld_path = str(Path(project_root) / "src" / "ansys_aedt" / "maxwell_2d" / "toy_model_1" / "flux_density.fld")
    config.analysis.grid_min = ["-120mm", "-120mm", "0mm"]
    config.analysis.grid_max = ["120mm", "120mm", "0mm"]
    config.analysis.grid_step = ["1mm", "1mm", "0mm"]

    print("[1/4] Tạo mô hình hình học 2D...")
    toy_model.create_model()

    print("[2/4] Chạy mô phỏng Ansys Maxwell...")
    toy_model.run_simulation()

    print("[3/4] Trích xuất & Parse dữ liệu vào toy_model.record.flux_density_field...")
    fld_output_path = toy_model.export_flux_density()

    print("[4/4] Vẽ heatmap mật độ từ thông trực tiếp từ toy_model.record...")
    toy_model.plot_flux_density(
        target_resolution=1500,
        sigma_smooth=0.0,
        vmin=0.0,
        vmax=2.0,
        figsize=(6.4, 4.8),
        dpi=100
    )

if __name__ == "__main__":
    main()
"""

# ==============================================================================
# 2. plot_flux_density.py (Dải B [0-2T], figsize tiêu chuẩn)
# ==============================================================================
PLOT_FLUX_DENSITY_CONTENT = """import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

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
    **kwargs
):
    \"\"\"
    Vẽ và hiển thị heatmap mật độ từ thông B với kích thước tiêu chuẩn (6.4x4.8 in, DPI=100) và dải B từ 0 đến 2 Tesla.
    \"\"\"
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

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    extent = [unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()]
    im = ax.imshow(
        grid_b_dense, 
        extent=extent, 
        origin='lower', 
        cmap='jet', 
        interpolation='bicubic',
        aspect='equal',
        vmin=vmin,
        vmax=vmax
    )

    cbar = fig.colorbar(im, ax=ax, label='|B| (Tesla)', pad=0.03)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('X (mm)', fontsize=9)
    ax.set_ylabel('Y (mm)', fontsize=9)

    ax.grid(False)
    plt.tight_layout()
    
    if show:
        print(f"🖼️ Đang hiển thị heatmap mật độ từ thông ({figsize[0]}x{figsize[1]} in, scale {vmin}-{vmax} T)...")
        plt.show()

    return fig
"""

# ==============================================================================
# 3. toy_model_1.py (Wrapper chuyển tiếp tham số)
# ==============================================================================
TOY_MODEL_1_CONTENT = """from types import SimpleNamespace
from .configuration import Configuration
from .create_model import create_model
from .run_simulation import run_simulation
from .export_flux_density import export_flux_density
from .plot_flux_density import plot_flux_density

class ToyModel1:
    def __init__(self):
        self.configuration = Configuration()
        self.m2d = None
        self.record = SimpleNamespace(
            flux_density_field=None
        )

    def create_model(self):
        return create_model(toy_model_obj=self)

    def run_simulation(self):
        return run_simulation(toy_model_obj=self)

    def export_flux_density(self, export_path=None, min_bounds=None, max_bounds=None, grid_step=None):
        return export_flux_density(
            toy_model_obj=self, 
            export_path=export_path,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            grid_step=grid_step
        )

    def plot_flux_density(
        self, 
        fld_path=None, 
        output_png_path=None,
        save_path=None,
        save_image_path=None,
        smooth="cubic", 
        target_resolution=1500, 
        sigma_smooth=0.0,
        show=True,
        figsize=(6.4, 4.8),
        dpi=100, 
        vmin=0.0,
        vmax=2.0,
        **kwargs
    ):
        return plot_flux_density(
            toy_model_obj=self,
            fld_path=fld_path, 
            output_png_path=output_png_path,
            save_path=save_path,
            save_image_path=save_image_path,
            smooth=smooth, 
            target_resolution=target_resolution, 
            sigma_smooth=sigma_smooth,
            show=show,
            figsize=figsize,
            dpi=dpi,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
"""

# Danh sách ánh xạ các file cần cập nhật đồng loạt
files_map = {
    scenarios_dir / "excute1.py": EXCUTE1_CONTENT,
    toy_model_dir / "plot_flux_density.py": PLOT_FLUX_DENSITY_CONTENT,
    toy_model_dir / "toy_model_1.py": TOY_MODEL_1_CONTENT,
}

# Ghi đè tự động và đồng loạt toàn bộ các file
for file_path, content in files_map.items():
  file_path.parent.mkdir(parents=True, exist_ok=True)
  with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
  print(f"✅ Đã ghi đè tự động: {file_path.relative_to(project_root)}")

print("\n🎉 ĐÃ HOÀN THÀNH CẬP NHẬT ĐỒNG LOẠT TOÀN BỘ DỰ ÁN!")