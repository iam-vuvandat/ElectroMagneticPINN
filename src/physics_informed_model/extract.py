import os

def generate_directory_tree(startpath):
    tree_str = []
    for root, dirs, files in os.walk(startpath):
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_str.append(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            if f.endswith('.py'):
                tree_str.append(f"{subindent}{f}")
                
    return '\n'.join(tree_str)

def generate_txt_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(base_dir, 'project_codebase.txt')
    current_script = os.path.basename(__file__)

    tree_content = generate_directory_tree(base_dir)

    txt_content = [
        "========================================",
        "DIRECTORY STRUCTURE",
        "========================================",
        tree_content,
        "",
        "========================================",
        "SOURCE CODE",
        "========================================"
    ]

    for root, _, files in os.walk(base_dir):
        if '__pycache__' in root:
            continue
        
        for file in sorted(files):
            if file.endswith('.py') and file != current_script:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                
                txt_content.append("")
                txt_content.append(f"--- FILE: {rel_path} ---")
                txt_content.append("")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    txt_content.append(f.read())
                    
                txt_content.append("")
                txt_content.append("="*40)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_content))

if __name__ == "__main__":
    generate_txt_file()