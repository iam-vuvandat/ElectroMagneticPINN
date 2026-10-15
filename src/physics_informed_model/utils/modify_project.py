import os
import sys

def modify_project():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))
    
    pinn_path = os.path.join(project_root, 'electro_magnetic_pinn.py')
    visualizer_path = os.path.join(project_root, 'utils', 'training_visualizer.py')
    test_sim_path = os.path.join(project_root, 'tests', 'test_simulation.py')

    print("[*] Bắt đầu cập nhật dự án để hỗ trợ xuất video MP4...")

    # =========================================================================
    # 1. CẬP NHẬT electro_magnetic_pinn.py
    # =========================================================================
    with open(pinn_path, 'r', encoding='utf-8') as f:
        pinn_code = f.read()

    old_vis_cfg = """        self.visualization_config = SimpleNamespace(
            active=False,
            update_interval=200,
            output_directory="animation_frames",
            resolution=80,
            gif_filename="training_process.gif",
            gif_fps=10
        )"""

    new_vis_cfg = """        self.visualization_config = SimpleNamespace(
            active=False,
            update_interval=200,
            output_directory="animation_frames",
            resolution=80,
            generate_gif_animation=True,
            gif_filename="training_process.gif",
            generate_mp4_video=True,
            mp4_filename="training_process.mp4",
            animation_fps=5,
            gif_fps=5,
            mp4_codec="libx264",
            mp4_quality=8
        )"""

    if old_vis_cfg in pinn_code:
        pinn_code = pinn_code.replace(old_vis_cfg, new_vis_cfg)
    else:
        print("[!] Không tìm thấy đoạn old_vis_cfg nguyên bản, bỏ qua thay thế visualization_config.")

    old_call = """        if self.visualizer_instance:
            print("Đang xuất file GIF Animation...")
            self.visualizer_instance.generate_gif()"""

    new_call = """        if self.visualizer_instance:
            self.visualizer_instance.generate_animations()"""

    if old_call in pinn_code:
        pinn_code = pinn_code.replace(old_call, new_call)

    with open(pinn_path, 'w', encoding='utf-8') as f:
        f.write(pinn_code)
    print(f"[✓] Đã cập nhật: {pinn_path}")

    # =========================================================================
    # 2. CẬP NHẬT utils/training_visualizer.py
    # =========================================================================
    with open(visualizer_path, 'r', encoding='utf-8') as f:
        vis_code = f.read()

    old_generate_gif_block = """    def generate_gif(self):
        output_filename = self.electro_magnetic_pinn.visualization_config.gif_filename
        fps = self.electro_magnetic_pinn.visualization_config.gif_fps
        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")"""

    new_animations_block = """    def generate_gif(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'gif_filename', 'training_process.gif')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps, loop=0)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")

    def generate_mp4(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        codec = getattr(cfg, 'mp4_codec', 'libx264')
        quality = getattr(cfg, 'mp4_quality', 8)

        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo MP4.")
            return

        valid_frames = [f for f in self.frame_paths if os.path.exists(f)]
        if not valid_frames:
            print("[!] CẢNH BÁO: Tất cả frame paths đều không tồn tại trên đĩa.")
            return

        full_mp4_path = os.path.join(self.output_directory, output_filename)
        print(f"[*] Đang xuất Video MP4 ({len(valid_frames)} frames, {fps} fps)...")
        
        exported = False
        try:
            writer = imageio.get_writer(
                full_mp4_path,
                fps=fps,
                codec=codec,
                pixelformat='yuv420p',
                quality=quality
            )
            for f in valid_frames:
                writer.append_data(imageio.imread(f))
            writer.close()
            exported = True
            print(f"[*] Đã xuất MP4 Video thành công (imageio) tại: {full_mp4_path}")
        except Exception as exc:
            print(f"[!] Không thể xuất MP4 bằng imageio-ffmpeg ({exc}). Chuyển sang OpenCV fallback...")

        if not exported:
            try:
                import cv2
                sample = cv2.imread(valid_frames[0])
                height, width, _ = sample.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(full_mp4_path, fourcc, float(fps), (width, height))
                for f in valid_frames:
                    img = cv2.imread(f)
                    video_writer.write(img)
                video_writer.release()
                print(f"[*] Đã xuất MP4 Video thành công (OpenCV) tại: {full_mp4_path}")
            except Exception as cv_exc:
                print(f"[!] Lỗi khi xuất video MP4 bằng OpenCV: {cv_exc}")

    def generate_animations(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        gen_gif = getattr(cfg, 'generate_gif_animation', True)
        gen_mp4 = getattr(cfg, 'generate_mp4_video', True)

        if gen_gif:
            self.generate_gif()
        if gen_mp4:
            self.generate_mp4()"""

    if old_generate_gif_block in vis_code:
        vis_code = vis_code.replace(old_generate_gif_block, new_animations_block)
    else:
        print("[!] Không tìm thấy hàm generate_gif cũ trong training_visualizer.py")

    with open(visualizer_path, 'w', encoding='utf-8') as f:
        f.write(vis_code)
    print(f"[✓] Đã cập nhật: {visualizer_path}")

    # =========================================================================
    # 3. CẬP NHẬT tests/test_simulation.py
    # =========================================================================
    with open(test_sim_path, 'r', encoding='utf-8') as f:
        test_code = f.read()

    old_test_vis = """    model.visualization_config.active = True
    model.visualization_config.update_interval = 200 
    model.visualization_config.output_directory = "figure"
    model.visualization_config.resolution = 80
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.gif_fps = 10"""

    new_test_vis = """    model.visualization_config.active = True
    model.visualization_config.update_interval = 50 
    model.visualization_config.output_directory = "figure"
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "training_process.mp4"
    model.visualization_config.animation_fps = 5
    model.visualization_config.gif_fps = 5"""

    if old_test_vis in test_code:
        test_code = test_code.replace(old_test_vis, new_test_vis)
    else:
        # Hỗ trợ thay thế nếu update_interval trong test_simulation đã được sửa sang 50 trước đó
        alt_old_test_vis = """    model.visualization_config.active = True
    model.visualization_config.update_interval = 50 
    model.visualization_config.output_directory = "figure"
    model.visualization_config.resolution = 80
    model.visualization_config.gif_filename = "training_process.gif"
    model.visualization_config.gif_fps = 10"""
        if alt_old_test_vis in test_code:
            test_code = test_code.replace(alt_old_test_vis, new_test_vis)

    with open(test_sim_path, 'w', encoding='utf-8') as f:
        f.write(test_code)
    print(f"[✓] Đã cập nhật: {test_sim_path}")

    print("[*] HOÀN TẤT MODIFY_PROJECT: Hệ thống đã hỗ trợ đầy đủ xuất cả GIF và MP4 qua Configuration!")

if __name__ == '__main__':
    modify_project()