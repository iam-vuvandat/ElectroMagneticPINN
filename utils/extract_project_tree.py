import os

def generate_tree(dir_path, prefix="", ignore_dirs=None, output_file="project_structure.txt"):
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', '.vscode', '.idea', 'venv', '.venv', 'build', 'data', 'dist','Ansys_Projects'}
    
    try:
        items = sorted(os.listdir(dir_path))
    except PermissionError:
        return

    items = [item for item in items if item not in ignore_dirs]
    
    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        line = f"{prefix}{connector}{item}"
        print(line)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_tree(path, new_prefix, ignore_dirs, output_file)

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_structure.txt")
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    root_name = os.path.basename(root_dir)
    header = f"{root_name}/"
    print(header)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(header + "\n")
        
    generate_tree(root_dir, output_file=output_file)