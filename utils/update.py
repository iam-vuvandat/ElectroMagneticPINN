import os

excute2_path = os.path.join("scenarios", "excute2.py")

if os.path.exists(excute2_path):
    with open(excute2_path, "r", encoding="utf-8") as f:
        content = f.read()

    config_flags = (
        "re_train_model = False\n"
        'checkpoint_name = "toymodel1"\n\n'
    )

    if "re_train_model =" not in content:
        target_insert_anchor = "# 1. QUY MÔ VẬT LÝ VÀ KHUNG TỌA ĐỘ (MET)"
        if target_insert_anchor in content:
            content = content.replace(target_insert_anchor, config_flags + target_insert_anchor)

    old_execution = (
        'model.update_electromagnetic_pinn()\n\n'
        'print("[*] Bắt đầu huấn luyện mô hình PINN với cấu hình Ansys FEM Colormap & Discrete Log Residual...")\n'
        'model.execute_training_process()\n\n'
        'print("[*] Đang lưu checkpoint mô hình với tên \'toymodel1\'...")\n'
        'model.save_checkpoint("toymodel1")'
    )

    new_execution = (
        'model.update_electromagnetic_pinn()\n\n'
        'if re_train_model:\n'
        '    print("[*] Starting PINN model training with Ansys FEM Colormap & Discrete Log Residual configuration...")\n'
        '    model.execute_training_process()\n'
        '    print(f"[*] Saving model checkpoint: \'{checkpoint_name}\'...")\n'
        '    model.save_checkpoint(checkpoint_name)\n'
        'else:\n'
        '    print(f"[*] Loading model checkpoint: \'{checkpoint_name}\'...")\n'
        '    model.load_checkpoint(checkpoint_name)'
    )

    if old_execution in content:
        content = content.replace(old_execution, new_execution)
    elif "model.execute_training_process()" in content:
        lines = content.splitlines()
        new_lines = []
        skip = False
        for line in lines:
            if "model.update_electromagnetic_pinn()" in line:
                new_lines.append(new_execution)
                skip = True
            elif skip:
                if 'print("[*] Xuất biểu đồ trường kết quả cuối cùng...")' in line:
                    skip = False
                    new_lines.append('print("[*] Exporting final field distribution plots...")')
            else:
                new_lines.append(line)
        content = "\n".join(new_lines)

    lines = content.splitlines()
    cleaned_lines = []
    for line in lines:
        if "====" in line:
            continue
        if "#" in line:
            line = line.split("#")[0]
        if line.strip() != "" or (cleaned_lines and cleaned_lines[-1].strip() != ""):
            cleaned_lines.append(line.rstrip())

    content = "\n".join(cleaned_lines)
    content = content.replace("[*] Bắt đầu huấn luyện mô hình PINN với cấu hình Ansys FEM Colormap & Discrete Log Residual...", "[*] Starting PINN model training with Ansys FEM Colormap & Discrete Log Residual configuration...")
    content = content.replace("[*] Xuất biểu đồ trường kết quả cuối cùng...", "[*] Exporting final field distribution plots...")
    content = content.replace("[✔] HOÀN THÀNH! Kết quả đã được lưu tại:", "[✔] COMPLETED! Results saved at:")

    with open(excute2_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully updated {excute2_path}")

else:
    print(f"File not found: {excute2_path}")