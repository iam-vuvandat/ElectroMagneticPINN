import os
import re
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

def simplify_background_sampling(base_dir):
    """1. Đơn giản hóa thuật toán lấy điểm không khí nền."""
    file_path = os.path.join(base_dir, "physics_domain", "collocation_sampler.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_fn = '''    def sample_background_interior_points(self, simulation_domain_object, number_of_points):
        if number_of_points <= 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        x_min, x_max = simulation_domain_object.x_boundaries_tuple
        y_min, y_max = simulation_domain_object.y_boundaries_tuple
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Lỗi lấy mẫu: Khung biên của SimulationDomain không hợp lệ.")

        # Lấy mẫu đồng đều (Sobol) toàn bộ miền không khí nền loại trừ phần thuộc vật liệu
        return self._sample_air_box(simulation_domain_object, number_of_points, x_min, x_max, y_min, y_max)'''

    pattern = r"    def sample_background_interior_points\(self, simulation_domain_object, number_of_points\):.*?(?=\n    def |\Z)"
    if re.search(pattern, content, flags=re.DOTALL):
        updated = re.sub(pattern, new_fn + "\n\n", content, flags=re.DOTALL)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f"[✔] Đã đơn giản hóa thuật toán lấy điểm không khí nền trong: {file_path}")

def update_visualizer_and_global_step(base_dir):
    """2. Cập nhật giao diện Subplot 1, nhãn trục hoành đồ thị Loss và tracking global_step."""
    tm_path = os.path.join(base_dir, "training_engine", "training_manager.py")
    if os.path.exists(tm_path):
        with open(tm_path, 'r', encoding='utf-8') as f:
            tm_code = f.read()
        if "self.global_step = 0" not in tm_code:
            tm_code = tm_code.replace(
                "self.completed_adam_epochs = 0",
                "self.completed_adam_epochs = 0\n        self.global_step = 0"
            )
            with open(tm_path, 'w', encoding='utf-8') as f:
                f.write(tm_code)
            print(f"[✔] Đã bổ sung self.global_step vào: {tm_path}")

    adam_path = os.path.join(base_dir, "training_engine", "train_adam.py")
    if os.path.exists(adam_path):
        with open(adam_path, 'r', encoding='utf-8') as f:
            adam_code = f.read()

        if "training_manager_instance.global_step += 1" not in adam_code:
            adam_code = adam_code.replace(
                "for epoch in range(epochs):",
                "for epoch in range(epochs):\n        training_manager_instance.global_step += 1"
            )

        old_call = "training_manager_instance.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)"
        new_call = "training_manager_instance.visualizer.save_frame(training_manager_instance.global_step, current_loss_value, domains_data)"
        if old_call in adam_code:
            adam_code = adam_code.replace(old_call, new_call)

        with open(adam_path, 'w', encoding='utf-8') as f:
            f.write(adam_code)
        print(f"[✔] Đã cập nhật global_step trong: {adam_path}")

    sf_path = os.path.join(base_dir, "training_visualizer", "save_frame.py")
    if os.path.exists(sf_path):
        with open(sf_path, 'r', encoding='utf-8') as f:
            sf_code = f.read()

        new_subplot1 = '''    # --- SUBPLOT 1: axs[0, 0] ---
    ax_pts = axs[0, 0]
    ax_pts.set_title("Categorized Collocation Points")
    
    domain_colors = ['#1a365d', '#742a2a', '#1c4532', '#4a5568', '#44337a', '#7b341e']
    pair_colors = ['#800000', '#004080', '#004d40', '#4a148c', '#bf360c']

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})

    for idx, (key, d) in enumerate(domains_dict.items()):
        c_dom = domain_colors[idx % len(domain_colors)]
        int_pts = d.get("interior_points")
        if int_pts is not None and int_pts.shape[0] > 0:
            pts_np = int_pts.detach().cpu().numpy()
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=0.5, color=c_dom, marker='o', linewidths=0, alpha=0.55)

    for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
        if_pts = pair_data.get("interface_points")
        if if_pts is not None and if_pts.shape[0] > 0:
            pts_np = if_pts.detach().cpu().numpy()
            c_pair = pair_colors[p_idx % len(pair_colors)]
            ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=1.0, color=c_pair, marker='o', linewidths=0, alpha=0.95)

    ax_pts.set_xlim(visualizer_instance.x_bounds)
    ax_pts.set_ylim(visualizer_instance.y_bounds)
    ax_pts.set_aspect('equal')'''

        pattern_sp1 = r"    # --- SUBPLOT 1: axs\[0, 0\] ---.*?ax_pts\.set_aspect\('equal'\)"
        if re.search(pattern_sp1, sf_code, flags=re.DOTALL):
            sf_code = re.sub(pattern_sp1, new_subplot1, sf_code, flags=re.DOTALL)

        sf_code = sf_code.replace('axs[0, 3].set_xlabel("Step / Epoch")', 'axs[0, 3].set_xlabel("Total Optimization Step")')
        sf_code = sf_code.replace('axs[0, 3].set_xlabel("Epoch")', 'axs[0, 3].set_xlabel("Total Optimization Step")')

        with open(sf_path, 'w', encoding='utf-8') as f:
            f.write(sf_code)
        print(f"[✔] Đã cập nhật Subplot 1 và nhãn Total Optimization Step trong: {sf_path}")

def add_colab_gif_display(base_dir):
    """3. Tự động hiển thị GIF tiến trình lên Colab sau khi huấn luyện xong."""
    file_path = os.path.join(base_dir, "training_visualizer", "generate_gif.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if "from IPython.display import Image, display" not in content:
        content = "from IPython.display import Image, display\n" + content

    display_code = """
    # Tự động hiển thị GIF trực tiếp trên Colab / Jupyter Cell Output
    try:
        if os.path.exists(full_gif_path):
            print("[*] Tự động hiển thị GIF tiến trình huấn luyện...")
            display(Image(filename=full_gif_path))
    except Exception as e:
        print(f"[!] Không thể hiển thị GIF trực tiếp: {e}")
"""

    if "display(Image(filename=full_gif_path))" not in content:
        if "imageio.mimsave" in content:
            pattern = r"(imageio\.mimsave\(.*?\))"
            content = re.sub(pattern, r"\1" + display_code, content, count=1)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã tích hợp tự động hiển thị GIF Colab vào: {file_path}")

def add_colab_frame_display_config(base_dir):
    """4. Thêm config display_colab_frames và cập nhật save_frame để in frame thời gian thực lên Colab."""
    # A. Cập nhật electro_magnetic_pinn.py
    pinn_path = os.path.join(base_dir, "electro_magnetic_pinn.py")
    if os.path.exists(pinn_path):
        with open(pinn_path, 'r', encoding='utf-8') as f:
            pinn_code = f.read()

        if "display_colab_frames" not in pinn_code:
            pinn_code = pinn_code.replace(
                "mp4_quality=8,",
                "mp4_quality=8,\n            display_colab_frames=False,"
            )
            with open(pinn_path, 'w', encoding='utf-8') as f:
                f.write(pinn_code)
            print(f"[✔] Đã thêm config display_colab_frames=False vào: {pinn_path}")

    # B. Cập nhật save_frame.py
    sf_path = os.path.join(base_dir, "training_visualizer", "save_frame.py")
    if os.path.exists(sf_path):
        with open(sf_path, 'r', encoding='utf-8') as f:
            sf_code = f.read()

        colab_live_snippet = """
    if getattr(vis_cfg, 'display_colab_frames', False):
        try:
            from IPython.display import Image, display, clear_output
            clear_output(wait=True)
            print(f"[*] Live Colab Frame at Step {epoch}: {frame_name}")
            display(Image(filename=frame_name))
        except Exception:
            pass
"""
        if "getattr(vis_cfg, 'display_colab_frames', False)" not in sf_code:
            sf_code = sf_code.rstrip() + "\n" + colab_live_snippet
            with open(sf_path, 'w', encoding='utf-8') as f:
                f.write(sf_code)
            print(f"[✔] Đã cập nhật tính năng hiển thị thời gian thực Colab trong: {sf_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Thực thi sửa đổi cấu hình dự án tại: {base_dir}\n")
    
    simplify_background_sampling(base_dir)
    update_visualizer_and_global_step(base_dir)
    add_colab_gif_display(base_dir)
    add_colab_frame_display_config(base_dir)
    
    print("\n[✔] HOÀN TẤT TẤT CẢ SỬA ĐỔI!")

if __name__ == "__main__":
    main()