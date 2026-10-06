import os
import re

def execute_figure_folder_fix():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # =====================================================================
    # 1. CẬP NHẬT TEST_SIMULATION.PY (Tạo và dọn dẹp thư mục figure ở đầu)
    # =====================================================================
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()
        
    # Thêm import shutil và logic dọn thư mục figure ngay đầu hàm main
    if "shutil.rmtree" not in test_content:
        test_content = test_content.replace(
            "def main():",
            "import shutil\n\ndef main():\n    if os.path.exists('figure'):\n        shutil.rmtree('figure')\n    os.makedirs('figure')"
        )
        
    # Cập nhật cấu hình thư mục đầu ra thành "figure"
    test_content = re.sub(
        r'model\.visualization_config\.output_directory\s*=\s*[\'"].*?[\'"]', 
        'model.visualization_config.output_directory = "figure"', 
        test_content
    )
    
    # Cập nhật ảnh kết quả cuối
    test_content = test_content.replace(
        "plt.savefig('final_results.png'", 
        "plt.savefig('figure/final_results.png'"
    )
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    # =====================================================================
    # 2. CẬP NHẬT GEOMETRY_VISUALIZER.PY (Lưu vào figure)
    # =====================================================================
    geom_file = os.path.join(project_root, 'geometry_engine', 'geometry_visualizer.py')
    with open(geom_file, 'r', encoding='utf-8') as f:
        geom_content = f.read()
        
    geom_content = geom_content.replace(
        "plt.savefig('geometry_plot.png'", 
        "plt.savefig('figure/geometry_plot.png'"
    )
    
    with open(geom_file, 'w', encoding='utf-8') as f:
        f.write(geom_content)

    # =====================================================================
    # 3. CẬP NHẬT TRAINING_VISUALIZER.PY (Lưu thẳng GIF vào figure, không xóa frame)
    # =====================================================================
    viz_file = os.path.join(project_root, 'utils', 'training_visualizer.py')
    with open(viz_file, 'r', encoding='utf-8') as f:
        viz_content = f.read()

    # Viết lại hoàn toàn hàm generate_gif (hàm cuối cùng trong file)
    new_generate_gif = '''    def generate_gif(self):
        output_filename = self.parent_pinn.visualization_config.gif_filename
        fps = self.parent_pinn.visualization_config.gif_fps
        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")'''

    # Thay thế hàm generate_gif cũ bằng nội dung mới
    viz_content = re.sub(r'    def generate_gif\(self\):.*', new_generate_gif, viz_content, flags=re.DOTALL)

    with open(viz_file, 'w', encoding='utf-8') as f:
        f.write(viz_content)

    print("[*] Đã cấu hình gom toàn bộ file (PNG tĩnh, GIF, Final, Geometry) vào chung thư mục 'figure'.")
    print("[*] Thư mục 'figure' sẽ bị xóa sạch và tạo mới lại tự động mỗi khi bắt đầu chạy file test_simulation.py.")

if __name__ == "__main__":
    execute_figure_folder_fix()