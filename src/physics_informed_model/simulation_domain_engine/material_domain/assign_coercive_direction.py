import math

def assign_coercive_direction(self, direction=(0, 1)):
    """
    Tính lại độ lớn lực kháng từ Hc theo Pythagoras từ Hc hiện tại,
    sau đó chuẩn hóa hướng truyền vào và cập nhật lại self.physics_config.coercive.
    """
    current_coercive = self.physics_config.coercive

    if callable(current_coercive):
        raise TypeError("Không thể áp dụng assign_coercive_direction khi coercive là một hàm/callable.")

    if isinstance(current_coercive, (list, tuple)):
        hc_x = float(current_coercive[0])
        hc_y = float(current_coercive[1])
    elif hasattr(current_coercive, 'tolist'):
        c_list = current_coercive.tolist()
        hc_x, hc_y = float(c_list[0]), float(c_list[1])
    else:
        raise ValueError("Kiểu dữ liệu của coercive không hợp lệ.")

    # 1. Tính cường độ Hc hiện tại bằng Pythagoras
    hc_magnitude = math.sqrt(hc_x**2 + hc_y**2)

    dir_x, dir_y = float(direction[0]), float(direction[1])
    dir_norm = math.sqrt(dir_x**2 + dir_y**2)

    if dir_norm <= 1e-12:
        raise ValueError("Vectơ hướng (direction) không được bằng 0.")

    # 2. Chuẩn hóa vectơ hướng
    unit_x = dir_x / dir_norm
    unit_y = dir_y / dir_norm

    # 3. Cập nhật lại thuộc tính coercive theo hướng mới
    self.physics_config.coercive = [hc_magnitude * unit_x, hc_magnitude * unit_y]

    print(f"[*] Domain '{self.domain_name}': Đã cập nhật hướng Hc thành {direction} với độ lớn |Hc| = {hc_magnitude:.2f} A/m.")
    return self
