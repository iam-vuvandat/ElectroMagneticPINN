import os

update_code = """import os

excute2_path = os.path.join("scenarios", "excute2.py")

if os.path.exists(excute2_path):
    with open(excute2_path, "r", encoding="utf-8") as f:
        content = f.read()

    target_sys_path = (
        "current_directory = os.path.dirname(os.path.abspath(__file__))\\n"
        "parent_directory = os.path.abspath(os.path.join(current_directory, '..'))\\n"
        "if parent_directory not in sys.path:\\n"
        "    sys.path.insert(0, parent_directory)\\n"
    )

    new_sys_path = (
        "current_directory = os.path.dirname(os.path.abspath(__file__))\\n"
        "parent_directory = os.path.abspath(os.path.join(current_directory, '..'))\\n"
        "pinn_module_path = os.path.join(parent_directory, 'src', 'physics_informed_model')\\n"
        "if parent_directory not in sys.path:\\n"
        "    sys.path.insert(0, parent_directory)\\n"
        "if pinn_module_path not in sys.path:\\n"
        "    sys.path.insert(0, pinn_module_path)\\n"
    )

    if target_sys_path in content:
        content = content.replace(target_sys_path, new_sys_path)
    elif "pinn_module_path not in sys.path" not in content:
        import_anchor = "import sys\\n"
        if import_anchor in content:
            content = content.replace(import_anchor, import_anchor + new_sys_path + "\\n")

    old_import = "from src.physics_informed_model.electro_magnetic_pinn import ElectroMagneticPINN"
    new_import = "from electro_magnetic_pinn import ElectroMagneticPINN"
    if old_import in content:
        content = content.replace(old_import, new_import)

    with open(excute2_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully fixed sys.path and imports in {excute2_path}")
else:
    print(f"File not found: {excute2_path}")
"""

with open("update.py", "w", encoding="utf-8") as f:
    f.write(update_code)

print("Created update.py")