import os
import inspect
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from scipy.interpolate import griddata

def parse_fld_file(fld_path: str):
    """Đọc và parse file dữ liệu .fld thành SimpleNamespace chứa dữ liệu mảng x, y, bx, by, b_mag."""
    if not os.path.exists(fld_path):
        raise FileNotFoundError(f"File dữ liệu .fld không tồn tại: {fld_path}")

    data = []
    with open(fld_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or any(line_str.startswith(h) for h in ["/", "Grid", "Unit", "Title", "Header", "Point"]):
                continue
            parts = [p.rstrip(",") for p in line_str.split()]
            if len(parts) >= 4:
                try:
                    row = [float(p) for p in parts]
                    data.append(row)
                except ValueError:
                    continue

    if not data:
        raise ValueError(f"File .fld rỗng hoặc chứa dữ liệu không hợp lệ: {fld_path}")

    data = np.array(data)
    x_mm = np.round(data[:, 0] * 1000.0, 4)  # Chuyển m -> mm
    y_mm = np.round(data[:, 1] * 1000.0, 4)  

    if data.shape[1] >= 6:
        bx = data[:, 3]
        by = data[:, 4]
        b_mag = np.sqrt(bx**2 + by**2)
    elif data.shape[1] == 4:
        bx = np.zeros_like(x_mm)
        by = np.zeros_like(y_mm)
        b_mag = data[:, 3]
    else:
        bx = data[:, 3]
        by = data[:, 4]
        b_mag = np.sqrt(bx**2 + by**2)

    unique_x = np.unique(x_mm)
    unique_y = np.unique(y_mm)

    X, Y = np.meshgrid(unique_x, unique_y)

    grid_bx = griddata((x_mm, y_mm), bx, (X, Y), method='nearest')
    grid_by = griddata((x_mm, y_mm), by, (X, Y), method='nearest')
    grid_b_mag = griddata((x_mm, y_mm), b_mag, (X, Y), method='nearest')

    return SimpleNamespace(
        x=unique_x,
        y=unique_y,
        X=X,
        Y=Y,
        bx=grid_bx,
        by=grid_by,
        b_mag=grid_b_mag,
        raw_data=data,
        fld_path=fld_path
    )

def export_flux_density(
    toy_model_obj, 
    export_path=None, 
    min_bounds=None, 
    max_bounds=None, 
    grid_step=None
):
    """Trích xuất từ trường từ Ansys Maxwell ra file .fld bằng method oModule.ExportOnGrid chuẩn từ AEDT record."""
    m2d = getattr(toy_model_obj, "m2d", None)
    config = toy_model_obj.configuration.analysis

    if export_path is None:
        export_path = config.b_fld_path
    if min_bounds is None:
        min_bounds = config.grid_min
    if max_bounds is None:
        max_bounds = config.grid_max
    if grid_step is None:
        grid_step = config.grid_step

    export_dir = os.path.dirname(export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    if m2d is not None:
        print(f"📊 Exporting B-field grid data to '{export_path}' via AEDT FieldsReporter...")
        try:
            o_fields_reporter = m2d.odesign.GetModule("FieldsReporter")
            o_fields_reporter.EnterQty("B")
            
            setup_str = getattr(m2d, "nominal_adaptive", None) or f"{getattr(config, 'setup_name', 'Setup1')} : LastAdaptive"
            
            export_options = [
                "NAME:ExportOption",
                "IncludePtInOutput:=", True,
                "RefCSName:=", "Global",
                "PtInSI:=", True,
                "FieldInRefCS:=", False
            ]

            o_fields_reporter.ExportOnGrid(
                export_path,
                min_bounds,
                max_bounds,
                grid_step,
                setup_str,
                [],
                export_options,
                "Cartesian",
                ["0mm", "0mm", "0mm"],
                False
            )
            print("✅ Đã trích xuất thành công bằng oModule.ExportOnGrid chuẩn Ansys!")
        except Exception as e:
            print(f"⚠️ Trích xuất qua oModule báo lỗi ({e}), chuyển sang gọi hàm wrapper PyAEDT post...")
            try:
                m2d.post.export_field_file_on_grid(
                    quantity="B",
                    grid_start=min_bounds,
                    grid_stop=max_bounds,
                    grid_step=grid_step,
                    filename=export_path
                )
            except Exception as e2:
                print(f"❌ Lỗi xuất dữ liệu: {e2}")
    else:
        print(f"⚠️ PyAEDT session m2d không khả dụng. Đang kiểm tra file sẵn có tại: '{export_path}'")

    field_data = parse_fld_file(export_path)
    if hasattr(toy_model_obj, "record"):
        toy_model_obj.record.flux_density_field = field_data
        print(f"✅ Đã đọc dữ liệu .fld và lưu thành công vào toy_model.record.flux_density_field")

    return export_path
