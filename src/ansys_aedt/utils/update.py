import os
from pathlib import Path

# Xác định vị trí thư mục mục tiêu
BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR / "maxwell_2d" / "toy_model_1"

if not TARGET_DIR.exists():
    if (BASE_DIR / "configuration.py").exists():
        TARGET_DIR = BASE_DIR
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {TARGET_DIR}, đang tự động tạo mới...")
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔄 Đang cập nhật các file tại: {TARGET_DIR}")

# 1. Nội dung file configuration.py
CONFIGURATION_CONTENT = """import os
from pathlib import Path
from types import SimpleNamespace

class Configuration:
    def __init__(self):
        self.name = "toy_model_1"
        self.version = "2025.2"
        self.solution_type = "Magnetostatic"
        
        # Kích thước hình học mm đồng bộ chuẩn với test_simulation.py
        self.geometry = SimpleNamespace(
            yoke=SimpleNamespace(
                x_start="-60mm",
                y_start="-30mm",
                width="120mm",
                height="30mm"
            ),
            magnet_left=SimpleNamespace(
                x_start="-50mm",
                y_start="0mm",
                width="45mm",
                height="18mm",
                coercive_direction=(0, 1)
            ),
            magnet_right=SimpleNamespace(
                x_start="5mm",
                y_start="0mm",
                width="45mm",
                height="18mm",
                coercive_direction=(0, -1)
            ),
            region=SimpleNamespace(
                x_start="-120mm",
                y_start="-120mm",
                width="240mm",
                height="240mm"
            )
        )
        
        self.materials = SimpleNamespace(
            yoke_material="steel_1008",
            magnet_material_left="NdFe30 - Copy",
            magnet_material_right="NdFe30 - Copy (2)",
            region_material="vacuum",
            hc_magnitude=838000.0  # A/m
        )

        default_b_fld_path = str(Path(__file__).resolve().parent / "flux_density.fld")

        self.analysis = SimpleNamespace(
            setup_name="Setup1",
            max_passes=10,
            min_passes=2,
            percent_error=1.0,
            max_element_length="1mm",
            b_fld_path=default_b_fld_path,
            grid_min=["-120mm", "-120mm", "0mm"],
            grid_max=["120mm", "120mm", "0mm"],
            grid_step=["1mm", "1mm", "0mm"]
        )
"""

# 2. Nội dung file export_flux_density.py
EXPORT_FLUX_DENSITY_CONTENT = """import os
from pathlib import Path

def export_flux_density(toy_model_obj, export_path=None, min_bounds=None, max_bounds=None, grid_step=None):
    print("\\033[94mIn function export_flux_density.\\033[0m")
    print("\\033[94m{\\033[0m")

    m2d = toy_model_obj.m2d
    if m2d is None:
        raise RuntimeError("Model chưa được khởi tạo! Hãy gọi create_model() trước khi xuất dữ liệu.")

    config = toy_model_obj.configuration
    
    if export_path is None:
        export_path = getattr(config.analysis, "b_fld_path", str(Path(__file__).resolve().parent / "flux_density.fld"))

    if min_bounds is None:
        min_bounds = getattr(config.analysis, "grid_min", ["-120mm", "-120mm", "0mm"])
    if max_bounds is None:
        max_bounds = getattr(config.analysis, "grid_max", ["120mm", "120mm", "0mm"])
    if grid_step is None:
        grid_step = getattr(config.analysis, "grid_step", ["1mm", "1mm", "0mm"])

    if os.path.exists(export_path):
        try:
            os.remove(export_path)
            print(f"\\033[94mIn function export_flux_density: Deleted existing file '{export_path}'.\\033[0m")
        except Exception as e:
            print(f"\\033[91mIn function export_flux_density: Failed to delete existing file '{export_path}': {e}\\033[0m")

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    o_module = m2d.odesign.GetModule("FieldsReporter")
    setup_solution = f"{config.analysis.setup_name} : LastAdaptive"

    o_module.ClearAllNamedExpr()
    o_module.EnterQty("B")

    o_module.ExportOnGrid(
        export_path,
        min_bounds,
        max_bounds,
        grid_step,
        setup_solution,
        [],
        [
            "NAME:ExportOption",
            "IncludePtInOutput:=", True,
            "RefCSName:=", "Global",
            "PtInSI:=", True,
            "FieldInRefCS:=", False
        ],
        "Cartesian",
        ["0mm", "0mm", "0mm"],
        False
    )

    print(f"\\033[94mIn function export_flux_density: B-field grid data exported successfully to: {export_path}\\033[0m")
    print("\\033[94m}\\033[0m\\n")

    return export_path
"""

# 3. Nội dung file plot_flux_density.py
PLOT_FLUX_DENSITY_CONTENT = """import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_flux_density(fld_path: str = None, save_path: str = None, title: str = "B-field Distribution (Tesla)"):
    \"\"\"
    Đọc dữ liệu file lưới .fld và hiển thị/lưu heatmap mật độ từ thông B bằng Matplotlib.
    \"\"\"
    if fld_path is None:
        fld_path = str(Path(__file__).resolve().parent / "flux_density.fld")

    if not os.path.exists(fld_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu lưới: {fld_path}")

    data = []
    with open(fld_path, "r") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("/") or line_str.startswith("Grid") or line_str.startswith("Unit") or line_str.startswith("Title"):
                continue
            parts = line_str.split()
            if len(parts) == 6:
                data.append([float(p) for p in parts])

    data = np.array(data)
    x = data[:, 0] * 1000.0  
    y = data[:, 1] * 1000.0  
    bx = data[:, 3]
    by = data[:, 4]
    b_mag = np.sqrt(bx**2 + by**2)

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    grid_b = b_mag.reshape(len(unique_x), len(unique_y)).T

    plt.figure(figsize=(9, 7))
    im = plt.imshow(
        grid_b,
        extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
        origin="lower",
        cmap="rainbow",
        vmin=0.0,
        vmax=2.0,
        aspect="equal"
    )
    cb = plt.colorbar(im, label="|B| (Tesla)")
    cb.ax.tick_params(labelsize=10)
    
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\\033[94mIn function plot_flux_density: Saved plot image to '{save_path}'.\\033[0m")
    else:
        plt.show()

    plt.close()
    return save_path
"""

# 4. Nội dung file toy_model_1.py
TOY_MODEL_1_CONTENT = """from .configuration import Configuration
from .create_model import create_model
from .run_simulation import run_simulation
from .export_flux_density import export_flux_density
from .plot_flux_density import plot_flux_density

class ToyModel1:
    def __init__(self):
        self.configuration = Configuration()
        self.m2d = None

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

    def plot_flux_density(self, fld_path=None, save_image_path=None):
        if fld_path is None:
            fld_path = getattr(self.configuration.analysis, "b_fld_path", None)
        return plot_flux_density(fld_path=fld_path, save_path=save_image_path)
"""

files_to_update = {
    TARGET_DIR / "configuration.py": CONFIGURATION_CONTENT,
    TARGET_DIR / "export_flux_density.py": EXPORT_FLUX_DENSITY_CONTENT,
    TARGET_DIR / "plot_flux_density.py": PLOT_FLUX_DENSITY_CONTENT,
    TARGET_DIR / "toy_model_1.py": TOY_MODEL_1_CONTENT,
}

for file_path, content in files_to_update.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Đã cập nhật: {file_path.name}")

print("\n✨ Tất cả các file đã được cập nhật thành công!")