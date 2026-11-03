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

def fix_evaluate_reluctivity(base_dir):
    """
    Sửa triệt để thuật toán nội suy trong evaluate_reluctivity.py:
    Sử dụng torch.bucketize(..., right=True) - 1 và xử lý mượt khoảng B -> 0 
    để không bị sập chỉ số âm về 0 làm phẳng nu_BH.
    """
    file_path = os.path.join(base_dir, "simulation_domain_engine", "material_domain", "evaluate_reluctivity.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    code_content = '''import torch
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

def evaluate_reluctivity(self, points_tensor, B_magnitude_tensor=None, is_nonlinear=None):
    if is_nonlinear is None:
        is_nonlinear = getattr(self.physics_config, 'is_nonlinear', False)

    if not is_nonlinear:
        mu_r = float(getattr(self.physics_config, 'relative_permeability', 1.0))
        const_nu = VACUUM_RELUCTIVITY / max(mu_r, 1e-6)
        return torch.full(
            (points_tensor.shape[0], 1),
            const_nu,
            dtype=points_tensor.dtype,
            device=points_tensor.device
        )

    bh_data = getattr(self.physics_config, 'bh_curve_data', None)
    if isinstance(bh_data, (list, tuple)) and len(bh_data) == 2:
        b_pts = torch.tensor(bh_data[0], dtype=points_tensor.dtype, device=points_tensor.device)
        h_pts = torch.tensor(bh_data[1], dtype=points_tensor.dtype, device=points_tensor.device)

        if B_magnitude_tensor is None:
            B_magnitude_tensor = torch.zeros(
                (points_tensor.shape[0], 1),
                dtype=points_tensor.dtype,
                device=points_tensor.device
            )

        b_flat = B_magnitude_tensor.view(-1)
        
        # Nội suy chính xác khoảng B_pts không bị tràn chỉ số âm
        indices = torch.bucketize(b_flat, b_pts, right=True) - 1
        indices = torch.clamp(indices, 0, len(b_pts) - 2)

        b0, b1 = b_pts[indices], b_pts[indices + 1]
        h0, h1 = h_pts[indices], h_pts[indices + 1]

        t = (b_flat - b0) / (b1 - b0 + 1e-12)
        h_interp = h0 + t * (h1 - h0)

        # Xử lý giới hạn khi B tiến sát 0
        nu_zero = h_pts[1] / max(b_pts[1].item(), 1e-12)
        nu_tensor = torch.where(b_flat > 1e-7, h_interp / torch.clamp(b_flat, min=1e-7), nu_zero)
        return nu_tensor.view(-1, 1)

    raise ValueError(f"Domain '{self.domain_name}' thiếu dữ liệu bh_curve_data hợp lệ.")
'''
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code_content)
    print(f"[✔] Đã sửa lỗi thuật toán nội suy B-H tại: {file_path}")

def fix_fit_nu_network_step(base_dir):
    """
    Sửa fit_nu_network_step.py: Đảm bảo pts_grad được truyền vào evaluate_reluctivity 
    và bật is_nonlinear=True cùng B_magnitude_tensor.
    """
    file_path = os.path.join(base_dir, "training_engine", "fit_nu_network_step.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_target_calc = "nu_BH_phys = domain.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_phys, is_nonlinear=True)"
    new_target_calc = "nu_BH_phys = domain.evaluate_reluctivity(pts_grad, B_magnitude_tensor=B_mag_phys, is_nonlinear=True)"

    if old_target_calc in content:
        content = content.replace(old_target_calc, new_target_calc)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã sửa luồng truyền tham số cho fit_nu_network_step tại: {file_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Đang thực thi sửa đổi logic (giữ nguyên material_data.py) tại: {base_dir}\n")
    
    fix_evaluate_reluctivity(base_dir)
    fix_fit_nu_network_step(base_dir)
    
    print("\n[✔] HOÀN TẤT SỬA LỖI! Tuyệt đối không can thiệp vào material_data.py.")

if __name__ == "__main__":
    main()