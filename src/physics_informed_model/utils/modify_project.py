import os

def fix_gif_generation_bugs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # =====================================================================
    # 1. SỬA LỖI TRONG TRAINING_VISUALIZER.PY (Thêm self.frame_paths.append)
    # =====================================================================
    viz_file = os.path.join(project_root, 'utils', 'training_visualizer.py')
    with open(viz_file, 'r', encoding='utf-8') as f:
        viz_content = f.read()
        
    old_save_end = """        frame_name = os.path.join(self.output_directory, f"frame_{epoch:05d}.png")
        plt.savefig(frame_name, dpi=100)
        plt.close(fig)
        
        # Đưa model quay lại chế độ train
        model.train()"""
        
    new_save_end = """        frame_name = os.path.join(self.output_directory, f"frame_{epoch:05d}.png")
        plt.savefig(frame_name, dpi=100)
        plt.close(fig)
        
        # BỔ SUNG: Lưu đường dẫn vào danh sách để tạo GIF
        self.frame_paths.append(frame_name)
        
        # Đưa model quay lại chế độ train
        model.train()"""
        
    viz_content = viz_content.replace(old_save_end, new_save_end)
    with open(viz_file, 'w', encoding='utf-8') as f:
        f.write(viz_content)

    # =====================================================================
    # 2. SỬA LỖI TRONG TRAINING_MANAGER.PY (Gắn camera cho L-BFGS)
    # =====================================================================
    training_file = os.path.join(project_root, 'training_manager.py')
    with open(training_file, 'r', encoding='utf-8') as f:
        training_content = f.read()
        
    old_lbfgs_closure = """            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
                
            if self.target_loss > 0 and loss.item() <= self.target_loss:"""
            
    new_lbfgs_closure = """            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
                
            # BỔ SUNG: Cho phép L-BFGS chụp ảnh định kỳ vào thư mục chung
            if self.visualizer and lbfgs_counter[0] % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(
                    50000 + lbfgs_counter[0], loss.item(), 
                    points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor
                )
                
            if self.target_loss > 0 and loss.item() <= self.target_loss:"""
            
    if "50000 + lbfgs_counter[0]" not in training_content:
        training_content = training_content.replace(old_lbfgs_closure, new_lbfgs_closure)
        
    with open(training_file, 'w', encoding='utf-8') as f:
        f.write(training_content)

    print("[*] Đã vá thành công lỗi thiếu `frame_paths.append` và tích hợp chụp ảnh cho L-BFGS!")

if __name__ == "__main__":
    fix_gif_generation_bugs()