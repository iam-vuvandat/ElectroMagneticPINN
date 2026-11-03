import os
import re
import sys

def get_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "..", "..")),
        "/content/ElectroMagneticPINN/src/physics_informed_model",
        "/content/ElectroMagneticPINN/physics_informed_model",
        os.getcwd()
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "training_visualizer")):
            return cand
    return os.path.abspath(os.path.join(script_dir, ".."))

def add_colab_gif_display(base_dir):
    """Bổ sung tính năng tự động hiển thị GIF trên Colab vào generate_gif.py."""
    file_path = os.path.join(base_dir, "training_visualizer", "generate_gif.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Thêm import IPython.display ở đầu file nếu chưa có
    if "from IPython.display import Image, display" not in content:
        content = "from IPython.display import Image, display\n" + content

    # 2. Đoạn mã tự động hiển thị GIF trên Notebook/Colab
    display_code = """
    # Tự động hiển thị GIF trực tiếp trên màn hình Colab / Jupyter Output
    try:
        if os.path.exists(full_gif_path):
            print("[*] Đang hiển thị GIF tiến trình huấn luyện lên Colab...")
            display(Image(filename=full_gif_path))
    except Exception as e:
        print(f"[!] Không thể hiển thị GIF trực tiếp: {e}")
"""

    # 3. Chèn đoạn mã hiển thị vào sau câu lệnh ghi GIF (mimsave hoặc print)
    if "display(Image(filename=full_gif_path))" not in content:
        if "imageio.mimsave" in content:
            pattern = r"(imageio\.mimsave\(.*?\))"
            content = re.sub(pattern, r"\1" + display_code, content, count=1)
        elif "print(f\"[✔] Đã tạo file GIF" in content or "print(f'[✔] Đã tạo file GIF" in content:
            pattern = r"(print\(f[\"']\[✔\] Đã tạo file GIF.*?[\"']\))"
            content = re.sub(pattern, r"\1\n" + display_code, content, count=1)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật tính năng tự động hiển thị GIF vào: {file_path}")
    else:
        print(f"[i] Tính năng hiển thị GIF đã được tích hợp sẵn trong {file_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Tiến hành cập nhật hiển thị GIF lên Colab tại: {base_dir}\n")
    add_colab_gif_display(base_dir)
    print("\n[✔] HOÀN TẤT! Sau khi kết thúc huấn luyện, file GIF sẽ tự động chạy trực tiếp trong cell output của Colab.")

if __name__ == "__main__":
    main()