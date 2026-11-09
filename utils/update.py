import os

def find_project_root():
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "scenarios", "excute2.py")):
        return cwd
    
    script_dir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(os.path.join(script_dir, "scenarios", "excute2.py")):
        return script_dir
        
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    if os.path.exists(os.path.join(parent_dir, "scenarios", "excute2.py")):
        return parent_dir
        
    return cwd

root_dir = find_project_root()
excute2_path = os.path.join(root_dir, "scenarios", "excute2.py")

if os.path.exists(excute2_path):
    with open(excute2_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_sys_code = (
        "current_directory = os.path.dirname(os.path.abspath(__file__))\n"
        "parent_directory = os.path.abspath(os.path.join(current_directory, '..'))\n"
        "if parent_directory not in sys.path:\n"
        "    sys.path.insert(0, parent_directory)"
    )

    new_sys_code = (
        "current_directory = os.path.dirname(os.path.abspath(__file__))\n"
        "parent_directory = os.path.abspath(os.path.join(current_directory, '..'))\n"
        "pinn_module_path = os.path.join(parent_directory, 'src', 'physics_informed_model')\n"
        "if parent_directory not in sys.path:\n"
        "    sys.path.insert(0, parent_directory)\n"
        "if pinn_module_path not in sys.path:\n"
        "    sys.path.insert(0, pinn_module_path)"
    )

    if target_sys_code in content:
        content = content.replace(target_sys_code, new_sys_code)
        with open(excute2_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully fixed sys.path in {excute2_path}")
    elif "pinn_module_path not in sys.path" in content:
        print(f"sys.path is already updated in {excute2_path}")
    else:
        print(f"Target block not matched exactly, updating via insertion in {excute2_path}")
        anchor = "if parent_directory not in sys.path:\n    sys.path.insert(0, parent_directory)"
        addition = (
            "\npinn_module_path = os.path.join(parent_directory, 'src', 'physics_informed_model')\n"
            "if pinn_module_path not in sys.path:\n"
            "    sys.path.insert(0, pinn_module_path)"
        )
        if anchor in content:
            content = content.replace(anchor, anchor + addition)
            with open(excute2_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Successfully appended pinn_module_path to {excute2_path}")
else:
    print(f"File not found: {excute2_path}")