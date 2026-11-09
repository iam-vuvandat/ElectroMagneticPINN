import os

root_dir = os.path.abspath(os.path.dirname(__file__))

# 1. SỬA LỖI SYS.PATH VÀ IMPORT TRONG scenarios/excute2.py
excute2_path = os.path.join(root_dir, "scenarios", "excute2.py")
if os.path.exists(excute2_path):
    with open(excute2_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Bổ sung pinn_module_path vào sys.path để các sub-modules (simulation_domain_engine, training_engine...) import thành công
    old_sys_setup = (
        "current_directory = os.path.dirname(os.path.abspath(__file__))\n"
        "parent_directory = os.path.abspath(os.path.join(current_directory, '..'))\n"
        "if parent_directory not in sys.path:\n"
        "    sys.path.insert(0, parent_directory)"
    )

    new_sys_setup = (
        "current_directory = os.path.dirname(os.path.abspath(__file__))\n"
        "parent_directory = os.path.abspath(os.path.join(current_directory, '..'))\n"
        "pinn_module_path = os.path.join(parent_directory, 'src', 'physics_informed_model')\n"
        "if parent_directory not in sys.path:\n"
        "    sys.path.insert(0, parent_directory)\n"
        "if pinn_module_path not in sys.path:\n"
        "    sys.path.insert(0, pinn_module_path)"
    )

    if old_sys_setup in content:
        content = content.replace(old_sys_setup, new_sys_setup)
    elif "pinn_module_path not in sys.path" not in content:
        # Trong trường hợp cấu trúc sys.path bị thay đổi nhẹ
        target_line = "if parent_directory not in sys.path:\n    sys.path.insert(0, parent_directory)"
        addition = (
            "\npinn_module_path = os.path.join(parent_directory, 'src', 'physics_informed_model')\n"
            "if pinn_module_path not in sys.path:\n"
            "    sys.path.insert(0, pinn_module_path)"
        )
        content = content.replace(target_line, target_line + addition)

    with open(excute2_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully updated sys.path in {excute2_path}")
else:
    print(f"File not found: {excute2_path}")


# 2. SỬA LỖI IMPORT SAI THƯ MỤC UTILS TRONG reluctance_synchronous_motor.py
rsm_file_path = os.path.join(
    root_dir,
    "src",
    "ansys_aedt",
    "rmxprt",
    "reluctane_synchronous_motor",
    "reluctance_synchronous_motor.py"
)

if os.path.exists(rsm_file_path):
    with open(rsm_file_path, "r", encoding="utf-8") as f:
        rsm_content = f.read()

    old_rsm_import = "from .utils.create_model import create_model"
    new_rsm_import = "from .create_model import create_model"

    if old_rsm_import in rsm_content:
        rsm_content = rsm_content.replace(old_rsm_import, new_rsm_import)
        with open(rsm_file_path, "w", encoding="utf-8") as f:
            f.write(rsm_content)
        print(f"Successfully fixed import in {rsm_file_path}")
    else:
        print(f"Import path already clean in {rsm_file_path}")
else:
    print(f"File not found: {rsm_file_path}")