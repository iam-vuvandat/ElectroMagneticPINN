import glob
import inspect
import os
import time

from ansys.aedt.core import Maxwell2d

# 1. Dọn dẹp process Ansys cũ
os.system("taskkill /F /IM ansysedt.exe /T")
os.system("taskkill /F /IM AnsysGRPC.exe /T")

ansoft_dir = os.path.expanduser(r"~\Documents\Ansoft")
for f in glob.glob(os.path.join(ansoft_dir, "*.aedt.auto")):
    try:
        os.remove(f)
    except Exception:
        pass
time.sleep(2)

# 2. Khởi tạo m2d cho Maxwell 2D (phiên bản 2025.2)
m2d = Maxwell2d(version="2025.2", new_desktop=True, non_graphical=False)


def list_all_members(obj, obj_name, file_handle, limit_depth=9999):
    file_handle.write(f"\n{'='*20} METHODS OF {obj_name} {'='*20}\n")

    for member in dir(obj):
        if member.startswith("_"):
            continue
        try:
            attr = getattr(obj, member)
            if callable(attr):
                sig = inspect.signature(attr)
                file_handle.write(f"{obj_name}.{member}{sig}\n")
        except Exception:
            continue

    if limit_depth > 0:
        try:
            children = {
                "modeler": obj.modeler,
                "materials": obj.materials,
                "boundaries": getattr(obj, "boundaries", None),
                "mesh": obj.mesh,
                "post": obj.post,
            }

            if obj_name == "m2d":
                for child_name, child_obj in children.items():
                    if child_obj is not None:
                        list_all_members(
                            child_obj,
                            f"m2d.{child_name}",
                            file_handle,
                            limit_depth - 1,
                        )
        except AttributeError:
            pass


output_path = os.path.join(os.getcwd(), "m2d_full_methods_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    list_all_members(m2d, "m2d", f)

print(
    f"[SUCCESS] Đã trích xuất xong toàn bộ method của Maxwell 2D vào:"
    f" {output_path}"
)

# Giải phóng session sau khi kiểm tra
m2d.release_desktop()