import os

def fix_numpy_cpu_error():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    test_file = os.path.join(project_root, 'test_simulation.py')
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()
        
    # Thêm .cpu() trước .numpy() cho các biến dự đoán đầu ra
    test_content = test_content.replace(
        "A_z_grid = A_z_pred.numpy()", 
        "A_z_grid = A_z_pred.cpu().numpy()"
    )
    test_content = test_content.replace(
        "B_x_grid = B_x_pred.numpy()", 
        "B_x_grid = B_x_pred.cpu().numpy()"
    )
    test_content = test_content.replace(
        "B_y_grid = B_y_pred.numpy()", 
        "B_y_grid = B_y_pred.cpu().numpy()"
    )
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
        
    print("[*] Đã sửa lỗi TypeError: Chuyển tensor từ GPU về CPU thành công trước khi dùng NumPy!")

if __name__ == "__main__":
    fix_numpy_cpu_error()