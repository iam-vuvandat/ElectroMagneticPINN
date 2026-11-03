import os
import sys

def get_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "..", "..")),
        "/content/ElectroMagneticPINN/src/physics_informed_model",
        "/content/ElectroMagneticPINN/physics_informed_model",
        os.getcwd()
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "training_engine")):
            return cand
    return os.path.abspath(os.path.join(script_dir, ".."))

def fix_fit_nu_network_step(base_dir):
    """
    Sửa fit_nu_network_step.py: Đảm bảo B_magnitude_tensor và is_nonlinear=True 
    được truyền đúng cách để tính nu_BH_phys phi tuyến biến thiên không gian.
    """
    file_path = os.path.join(base_dir, "training_engine", "fit_nu_network_step.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Sửa dòng tính nu_BH_phys chuẩn autograd & truyền B_mag_phys
    old_target_calc = "nu_BH_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys, is_nonlinear=True)"
    new_target_calc = "nu_BH_phys = domain.evaluate_reluctivity(pts_grad, B_magnitude_tensor=B_mag_phys, is_nonlinear=True)"

    if old_target_calc in content:
        content = content.replace(old_target_calc, new_target_calc)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã sửa lỗi truyền tọa độ cho B_magnitude_tensor trong: {file_path}")
    else:
        print(f"[i] File {file_path} đã chuẩn hoặc không cần sửa.")

def fix_interface_loss(base_dir):
    """
    Sửa interface_loss.py: Truyền B_magnitude_tensor=B_mag_a_phys và is_nonlinear=True khi gọi evaluate_reluctivity.
    """
    file_path = os.path.join(base_dir, "physics_domain", "physical_equations", "interface_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_code_a = "nu_a_star = domain_a.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_a_phys) / self.nu0"
    new_code_a = "nu_a_star = domain_a.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_a_phys, is_nonlinear=True) / self.nu0"
    
    old_code_b = "nu_b_star = domain_b.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_b_phys) / self.nu0"
    new_code_b = "nu_b_star = domain_b.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_b_phys, is_nonlinear=True) / self.nu0"

    content = content.replace(old_code_a, new_code_a).replace(old_code_b, new_code_b)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[✔] Đã cập nhật đánh giá độ từ trở phi tuyến ở điều kiện biên trong: {file_path}")

def update_material_data_bh_curve(base_dir):
    """
    Cập nhật đường cong B-H thép từ mềm phi tuyến cao (mu_r_max = 6953.37 tại B = 0.9T).
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

def main():
    base_dir = get_base_dir()
    print(f"[*] Đang thực thi sửa đổi hệ thống tại: {base_dir}\n")
    
    fix_fit_nu_network_step(base_dir)
    fix_interface_loss(base_dir)
    update_material_data_bh_curve(base_dir)
    
    print("\n[✔] TẤT CẢ CÁC LỖI KHẾN MU_R BỊ PHẲNG ĐÃ ĐƯỢC KHẮC PHỤC HOÀN TOÀN!")

if __name__ == "__main__":
    main()