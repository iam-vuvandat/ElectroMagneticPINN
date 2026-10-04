import os

def generate_directory_tree(start_directory_path):
    tree_string_list = []
    for root_directory, directories_list, files_list in os.walk(start_directory_path):
        if '__pycache__' in directories_list:
            directories_list.remove('__pycache__')
        if '.git' in directories_list:
            directories_list.remove('.git')
        if '.venv' in directories_list:
            directories_list.remove('.venv')
            
        level_count = root_directory.replace(start_directory_path, '').count(os.sep)
        indentation_string = ' ' * 4 * level_count
        tree_string_list.append(f"{indentation_string}{os.path.basename(root_directory)}/")
        
        sub_indentation_string = ' ' * 4 * (level_count + 1)
        for file_name in sorted(files_list):
            if file_name.endswith('.py'):
                tree_string_list.append(f"{sub_indentation_string}{file_name}")
                
    return '\n'.join(tree_string_list)

def extract_physics_informed_model_source_code():
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    
    target_directory_path = os.path.abspath(os.path.join(current_script_directory, '..'))
    
    output_file_path = os.path.join(current_script_directory, 'physics_model_source_code.txt')
    current_script_name = os.path.basename(__file__)

    directory_tree_content = generate_directory_tree(target_directory_path)

    text_content_list = [
        "========================================",
        "DIRECTORY STRUCTURE",
        "========================================",
        directory_tree_content,
        "",
        "========================================",
        "SOURCE CODE",
        "========================================"
    ]

    for root_directory, directories_list, files_list in os.walk(target_directory_path):
        if '__pycache__' in root_directory or '.git' in root_directory or '.venv' in root_directory:
            continue
        
        for file_name in sorted(files_list):
            if file_name.endswith('.py') and file_name != current_script_name:
                absolute_file_path = os.path.join(root_directory, file_name)
                relative_file_path = os.path.relpath(absolute_file_path, target_directory_path)
                
                text_content_list.append("")
                text_content_list.append(f"--- FILE: {relative_file_path} ---")
                text_content_list.append("")
                
                try:
                    with open(absolute_file_path, 'r', encoding='utf-8') as file_object:
                        lines_list = file_object.readlines()
                        for line_string in lines_list:
                            if line_string.startswith("if __name__ == '__main__':") or line_string.startswith('if __name__ == "__main__":'):
                                break
                            text_content_list.append(line_string.rstrip('\n'))
                except Exception as exception_object:
                    text_content_list.append(f"Error reading file: {exception_object}")
                    
                text_content_list.append("")
                text_content_list.append("="*40)

    with open(output_file_path, 'w', encoding='utf-8') as output_file_object:
        output_file_object.write('\n'.join(text_content_list))

if __name__ == "__main__":
    extract_physics_informed_model_source_code()