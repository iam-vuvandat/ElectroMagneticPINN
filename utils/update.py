from pathlib import Path
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

    # --- CẤU HÌNH PHÂN TÍCH & TRÍCH XUẤT LƯỚI (ANALYSIS) ---
    config.analysis.setup_name = "Setup1"
    config.analysis.max_passes = 10
    config.analysis.min_passes = 2
    config.analysis.percent_error = 1.0
    config.analysis.max_element_length = "0.025mm"
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

    print("[4/4] Vẽ heatmap mật độ từ thông trực tiếp (Kế thừa 20 màu ANSYS FEM từ PINN)...")
    toy_model.plot_flux_density(
        target_resolution=1500,
        sigma_smooth=0.0,
        vmin=0.0,
        vmax=2.0,
        cmap="ansys_fem",
        figsize=(6.4, 4.8),
        dpi=100
    )

if __name__ == "__main__":
    main()