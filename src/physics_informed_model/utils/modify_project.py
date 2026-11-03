import os
import sys

def get_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.join(script_dir, "physics_informed_model"),
        os.getcwd()
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "training_engine")):
            return cand
    return os.path.abspath(os.path.join(script_dir, ".."))


def fix_compute_loss(base_dir):
    """
    Sửa compute_loss.py: Giữ nguyên kết nối autograd với pts để tính \nabla \nu(x,y)
    trong phương trình Maxwell, nhưng không đưa Net_nu vào Adam.
    """
    file_path = os.path.join(base_dir, "training_engine", "compute_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_code = "nu_star_net = nu_star_pred.detach() if nu_star_pred is not None else None"
    new_code = "# Giữ nguyên đồ thị autograd để tính toán đạo hàm không gian \nabla \nu(x,y) trong Maxwell PDE\n            nu_star_net = nu_star_pred if nu_star_pred is not None else None"

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã phục hồi đạo hàm không gian \\nabla \\nu(x,y) trong: {file_path}")
    else:
        print(f"[!] File {file_path} đã được sửa hoặc không tìm thấy chuỗi khớp.")


def update_material_data_bh_curve(base_dir):
    """
    Cập nhật đường cong B-H thép từ mềm chất lượng cao (mu_r_max = 6953.37 tại B = 0.9T)
    để tạo sự phân hóa phi tuyến mạnh mẽ trong miền từ thông.
    """
    file_path = os.path.join(base_dir, "simulation_domain_engine", "material_library", "material_data.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Thay thế đường cong steel_1008 bằng đường cong phi tuyến chất lượng cao
    old_steel_block = """            SimpleNamespace(
                name="steel_1008",
                relative_permeability=1000.0,  
                coercive=[0.0, 0.0],
                bh_curve_data=[
                    # B_points (Tesla)
                    [0.0, 0.2402, 0.8654, 1.1106, 1.2458, 1.331, 1.5, 1.6, 1.683, 
                     1.741, 1.78, 1.905, 2.025, 2.085, 2.13, 2.165, 2.28, 2.485, 2.5851],
                    # H_points (A/m)
                    [0.0, 159.2, 318.3, 477.5, 636.6, 795.8, 1591.5, 3183.1, 4774.6, 
                     6366.2, 7957.7, 15915.5, 31831.0, 47746.5, 63662.0, 79577.5, 
                     159155.0, 318310.0, 397887.0]
                ]
            )"""

    new_steel_block = """            SimpleNamespace(
                name="steel_1008",
                relative_permeability=1000.0,  
                coercive=[0.0, 0.0],
                bh_curve_data=[
                    # B_points (Tesla) - Thép từ mềm phi tuyến cao (mu_r_max = 6953.37 tại 0.9T)
                    [0.0, 0.1, 0.5, 0.8, 0.9, 1.0, 1.3, 1.5, 1.9, 2.4],
                    # H_points (A/m)
                    [0.0, 34.8, 67.4, 91.8, 103.0, 119.0, 250.0, 1180.0, 25000.0, 229580.0]
                ]
            )"""

    if old_steel_block in content:
        content = content.replace(old_steel_block, new_steel_block)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật bảng B-H phi tuyến cao cho steel_1008 trong: {file_path}")
    else:
        print(f"[!] Bảng B-H trong {file_path} đã được cập nhật trước đó.")


def update_nu_network_initialization(base_dir):
    """
    Sửa xpinn_architecture.py: Tăng gain khởi tạo mạng Net_nu để phản ứng nhạy với độ dốc không gian.
    """
    file_path = os.path.join(base_dir, "architectures", "xpinn_architecture.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_init = "nn.init.xavier_normal_(module.weight, gain=0.10)"
    new_init = "nn.init.xavier_normal_(module.weight, gain=0.50)  # Tăng gain để học độ dốc không gian tốt hơn"

    if old_init in content:
        content = content.replace(old_init, new_init)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã tối ưu hóa khởi tạo trọng số Net_nu trong: {file_path}")


def main():
    base_dir = get_base_dir()
    print(f"[*] Đang thực thi nâng cấp tính phi tuyến vật lý tại: {base_dir}\n")
    
    fix_compute_loss(base_dir)
    update_material_data_bh_curve(base_dir)
    update_nu_network_initialization(base_dir)
    
    print("\n[✔] TẤT CẢ CÁC ĐIỂM NGHẼN TÍNH PHI TUYẾN ĐÃ ĐƯỢC XỬ LÝ HOÀN TẢO!")

if __name__ == "__main__":
    main()