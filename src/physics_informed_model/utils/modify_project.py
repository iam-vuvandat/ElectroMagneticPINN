import os

def execute_gif_cleanup_optimization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # =====================================================================
    # 1. CẬP NHẬT THƯ MỤC LƯU TRONG TEST_SIMULATION.PY
    # =====================================================================
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()
        
    # Đổi tên thư mục từ "animation_frames" sang "animation"
    test_content = test_content.replace(
        'model.visualization_config.output_directory = "animation_frames"', 
        'model.visualization_config.output_directory = "animation"'
    )
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    # =====================================================================
    # 2. CẬP NHẬT HÀM GENERATE_GIF (LƯU ĐÚNG CHỖ & TỰ ĐỘNG XÓA ẢNH TĨNH)
    # =====================================================================
    visualizer_file = os.path.join(project_root, 'utils', 'training_visualizer.py')
    with open(visualizer_file, 'r', encoding='utf-8') as f:
        viz_content = f.read()

    old_generate_gif = """    def generate_gif(self):
        output_filename = self.parent_pinn.visualization_config.gif_filename
        fps = self.parent_pinn.visualization_config.gif_fps
        if not self.frame_paths:
            return
        images = []
        for filename in self.frame_paths:
            images.append(imageio.imread(filename))
        imageio.mimsave(output_filename, images, fps=fps)"""

    new_generate_gif = """    def generate_gif(self):
        output_filename = self.parent_pinn.visualization_config.gif_filename
        fps = self.parent_pinn.visualization_config.gif_fps
        if not self.frame_paths:
            return
            
        # Ghép đường dẫn để lưu GIF vào bên trong thư mục animation/
        full_gif_path = os.path.join(self.output_directory, output_filename)
        
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        # Xuất file GIF
        imageio.mimsave(full_gif_path, images, fps=fps)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")
        
        # Dọn dẹp (xóa) các frame tĩnh sau khi tạo GIF thành công
        print("[*] Đang dọn dẹp các frame tĩnh...")
        cleaned_count = 0
        for filename in self.frame_paths:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    cleaned_count += 1
            except Exception as e:
                pass
        print(f"[*] Đã xóa dọn dẹp {cleaned_count} ảnh PNG tĩnh.")"""

    viz_content = viz_content.replace(old_generate_gif, new_generate_gif)

    with open(visualizer_file, 'w', encoding='utf-8') as f:
        f.write(viz_content)

    print("[*] Cập nhật thành công! GIF sẽ được lưu vào thư mục 'animation/' và các ảnh PNG sẽ tự động bị xóa.")

if __name__ == "__main__":
    execute_gif_cleanup_optimization()