import os
import re

def modify_training_manager(file_path):
    if not os.path.exists(file_path):
        print(f"[-] Không tìm thấy file: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Sửa lỗi .detach() làm đứt chuỗi đạo hàm không gian của Net_nu
    old_pattern = r"nu_star_net\s*=\s*nu_star_pred\.detach\(\)\s*if\s*nu_star_pred\s*is\s*not\s*None\s*else\s*None"
    new_code = "nu_star_net = nu_star_pred if nu_star_pred is not None else None"

    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_code, content)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã sửa lỗi .detach() trong: {file_path}")
        return True
    else:
        print(f"[!] Không tìm thấy đoạn code nu_star_pred.detach() hoặc đã được sửa trước đó trong: {file_path}")
        return False


def modify_test_simulation(file_path):
    if not os.path.exists(file_path):
        print(f"[-] Không tìm thấy file: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Bật Fourier Features cho nu_network_config của iron_yoke
    content = content.replace(
        "iron_yoke.nu_network_config.fourier_features = False",
        "iron_yoke.nu_network_config.fourier_features = True\n    iron_yoke.nu_network_config.fourier_sigma = 1.0\n    iron_yoke.nu_network_config.fourier_mapping_size = 64"
    )

    # 2. Tăng tham số fit_nu_epochs và fit_nu_lr
    content = content.replace("model.training_config.fit_nu_epochs = 100", "model.training_config.fit_nu_epochs = 200")
    content = content.replace("model.training_config.fit_nu_lr = 2e-3", "model.training_config.fit_nu_lr = 4e-3")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[✔] Đã cập nhật cấu hình Net_nu và Fourier Features trong: {file_path}")
    return True


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    training_manager_path = os.path.join(base_dir, "training_engine", "training_manager.py")
    test_sim_path = os.path.join(base_dir, "tests", "test_simulation.py")

    print("[*] Đang thực thi chỉnh sửa dự án...")
    modify_training_manager(training_manager_path)
    modify_test_simulation(test_sim_path)
    print("[*] Hoàn tất!")


if __name__ == "__main__":
    main()