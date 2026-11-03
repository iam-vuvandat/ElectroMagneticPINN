import os
import sys

def get_base_dir():
    """
    Xác định thư mục gốc 'physics_informed_model' một cách linh hoạt.
    """
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

def fix_training_manager(base_dir):
    """
    Sửa training_manager.py: Loại bỏ nu_networks khỏi Adam chính.
    """
    file_path = os.path.join(base_dir, "training_engine", "training_manager.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_code = "all_params = list(model.networks.parameters()) + (list(model.nu_networks.parameters()) if hasattr(model, 'nu_networks') else [])\n        self.optimizer_adam = optim.Adam(all_params, lr=self.base_learning_rate_adam)"
    new_code = "# Chỉ tối ưu hóa tham số của Net_Az trong Adam. Net_nu được cập nhật riêng qua bước FPI (fit_nu_network_step)\n        self.optimizer_adam = optim.Adam(model.networks.parameters(), lr=self.base_learning_rate_adam)"

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã sửa lỗi trôi trọng số Net_nu trong Adam tại: {file_path}")
    else:
        print(f"[!] Đoạn mã trong {file_path} đã được cập nhật hoặc không khớp chuỗi tìm kiếm.")

def fix_compute_loss(base_dir):
    """
    Sửa compute_loss.py: Ngắt đồ thị đạo hàm (detach) từ PDE Loss về Net_nu trong Stage 1A.
    """
    file_path = os.path.join(base_dir, "training_engine", "compute_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_code = """            if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                nu_star_pred = model.forward_nu_network(net_key, pts)
                nu_star_net = nu_star_pred if nu_star_pred is not None else None"""

    new_code = """            if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                nu_star_pred = model.forward_nu_network(net_key, pts)
                # Detach nu_star_pred để ngăn gradient PDE làm nhiễu Net_nu trong lúc luyện Net_Az
                nu_star_net = nu_star_pred.detach() if nu_star_pred is not None else None"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cô lập đồ thị đạo hàm (detach) của Net_nu tại: {file_path}")
    else:
        print(f"[!] Đoạn mã trong {file_path} đã được cập nhật hoặc không khớp chuỗi tìm kiếm.")

def update_save_frame_file(base_dir):
    """
    Đảm bảo file visualizer save_frame.py được đồng bộ chính xác.
    """
    vis_dir = os.path.join(base_dir, "training_visualizer")
    os.makedirs(vis_dir, exist_ok=True)
    file_path = os.path.join(vis_dir, "save_frame.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Kiểm tra xem label colorbar có bị lỗi gạch chéo kép (\\mu_r) không và sửa lại
    if r"\\mu_r" in content:
        content = content.replace(r"\\mu_r", r"\mu_r")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã sửa nhãn hiển thị LaTeX trong: {file_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Đang thực thi chỉnh sửa dự án tại thư mục gốc: {base_dir}\n")
    
    fix_training_manager(base_dir)
    fix_compute_loss(base_dir)
    update_save_frame_file(base_dir)
    
    print("\n[✔] TẤT CẢ CÁC ĐIỂM SỬA VẬT LÝ VÀ ĐỒ THỊ ĐÃ ĐƯỢC CẬP NHẬT THÀNH CÔNG!")

if __name__ == "__main__":
    main()