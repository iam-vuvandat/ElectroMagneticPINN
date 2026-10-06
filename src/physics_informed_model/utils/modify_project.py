import os

def clean_all_lbfgs_configs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        # Nếu dòng nào chứa chữ 'lbfgs' thì tiến hành comment lại
        if 'lbfgs' in line.lower():
            new_lines.append(f"    # {line.strip()} # Đã tự động loại bỏ\n")
        else:
            new_lines.append(line)
            
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"[*] Đã dọn dẹp sạch sẽ các tham số L-BFGS trong: {test_file}")

if __name__ == "__main__":
    clean_all_lbfgs_configs()