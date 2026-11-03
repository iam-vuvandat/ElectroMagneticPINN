import os
import sys

def get_base_dir():
    """
    Xác định thư mục gốc 'physics_informed_model' linh hoạt trên Colab và Local.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "..", "..")),
        cwd,
        os.path.join(cwd, "src", "physics_informed_model"),
        "/content/ElectroMagneticPINN/src/physics_informed_model",
        "/content/ElectroMagneticPINN/physics_informed_model",
        "/content/ElectroMagneticPINN"
    ]
    
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "training_engine")) and os.path.isdir(os.path.join(cand, "architectures")):
            return cand
            
    return os.path.abspath(os.path.join(script_dir, ".."))


def fix_compute_loss(base_dir):
    """
    Sửa compute_loss.py: Giữ nguyên đồ thị autograd cho nu_star_pred để tính đạo hàm grad(nu)
    trong Maxwell PDE mà không làm lỗi cú pháp Python.
    """
    file_path = os.path.join(base_dir, "training_engine", "compute_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Trường hợp 1: Sửa file nếu đang bị dính lỗi syntax line un-comment
    if "u(x,y) trong Maxwell PDE" in content or "nu_star_net = nu_star_pred.detach()" in content:
        # Thay thế khối mã bị lỗi/cũ bằng khối mã chuẩn autograd
        lines = content.splitlines()
        new_lines = []
        skip_bad_line = False
        for line in lines:
            if "u(x,y) trong Maxwell PDE" in line:
                continue
            if "nu_star_net = nu_star_pred.detach() if nu_star_pred is not None else None" in line:
                indent = line[:line.find("nu_star_net")]
                new_lines.append(f"{indent}# Giu nguyen do thi autograd de tinh dao ham khong gian grad(nu) trong Maxwell PDE")
                new_lines.append(f"{indent}nu_star_net = nu_star_pred if nu_star_pred is not None else None")
            else:
                new_lines.append(line)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(new_lines))
        print(f"[✔] Đã sửa triệt để lỗi SyntaxError và phục hồi autograd grad(nu) tại: {file_path}")
    else:
        print(f"[i] File {file_path} đã ở trạng thái chuẩn.")


def update_material_data_bh_curve(base_dir):
    """
    Cập nhật đường cong B-H thép từ mềm phi tuyến (mu_r_max ~ 6953 tại B = 0.9T).
    """
    file_path = os.path.join(base_dir, "simulation_domain_engine", "material_library", "material_data.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

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
                    # B_points (Tesla) - Du lieu phi tuyen cao (mu_r_max = 6953.37 tai B = 0.9T)
                    [0.0, 0.1, 0.5, 0.8, 0.9, 1.0, 1.3, 1.5, 1.9, 2.4],
                    # H_points (A/m)
                    [0.0, 34.8, 67.4, 91.8, 103.0, 119.0, 250.0, 1180.0, 25000.0, 229580.0]
                ]
            )"""

    if old_steel_block in content:
        content = content.replace(old_steel_block, new_steel_block)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật bảng B-H phi tuyến cho steel_1008 tại: {file_path}")


def update_nu_network_initialization(base_dir):
    """
    Tăng gain khởi tạo mạng Net_nu trong xpinn_architecture.py.
    """
    file_path = os.path.join(base_dir, "architectures", "xpinn_architecture.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_init = "nn.init.xavier_normal_(module.weight, gain=0.10)"
    new_init = "nn.init.xavier_normal_(module.weight, gain=0.50)"

    if old_init in content:
        content = content.replace(old_init, new_init)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã tối ưu hóa khởi tạo trọng số Net_nu tại: {file_path}")


def main():
    base_dir = get_base_dir()
    print(f"[*] Đang thực thi sửa đổi dự án tại: {base_dir}\n")
    
    fix_compute_loss(base_dir)
    update_material_data_bh_curve(base_dir)
    update_nu_network_initialization(base_dir)
    
    print("\n[✔] HOÀN TẤT SỬA LỖI SYNTAX VÀ CẬP NHẬT MÃ NGUỒN!")

if __name__ == "__main__":
    main()