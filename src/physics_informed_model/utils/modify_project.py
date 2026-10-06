import os

def update_steepness_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))

    # =====================================================================
    # 1. CẬP NHẬT TỆP: geometry_engine/segment/segment.py
    # =====================================================================
    segment_file = os.path.join(project_root, 'geometry_engine', 'segment', 'segment.py')
    with open(segment_file, 'r', encoding='utf-8') as f:
        content = f.read()

    old_init = """    def __init__(self, 
                 outline=None,
                 material="air",
                 relative_permeability=1.0,
                 bh_curve=None,
                 coercive=[0.0, 0.0],
                 current=0.0):
        
        self.material = material
        self.relative_permeability = relative_permeability
        self.coercive = coercive
        self.current = current
        self.bh_curve = bh_curve
        
        self.outline = None
        self.outline_tensor = None
        self.section_area = 0.0
        self.current_density = 0.0
        self.steepness = 1.0 
        
        if outline is not None:
            self.set_outline(outline) 
            self.compute_current_density()
            self.calculate_penetrating_steepness()"""

    new_init = """    def __init__(self, 
                 outline=None,
                 material="air",
                 relative_permeability=1.0,
                 bh_curve=None,
                 coercive=[0.0, 0.0],
                 current=0.0,
                 steepness=None): # BỔ SUNG: Nhận tham số steepness tùy chỉnh
        
        self.material = material
        self.relative_permeability = relative_permeability
        self.coercive = coercive
        self.current = current
        self.bh_curve = bh_curve
        
        self.outline = None
        self.outline_tensor = None
        self.section_area = 0.0
        self.current_density = 0.0
        
        # BỔ SUNG: Kiểm tra xem người dùng có gán cứng steepness hay không
        self.steepness = 1.0 if steepness is None else steepness
        self._has_custom_steepness = steepness is not None 
        
        if outline is not None:
            self.set_outline(outline) 
            self.compute_current_density()
            # Chỉ tự động tính steepness nếu người dùng KHÔNG truyền vào
            if not self._has_custom_steepness:
                self.calculate_penetrating_steepness()"""

    if "steepness=None" not in content:
        content = content.replace(old_init, new_init)
        with open(segment_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[*] Đã cập nhật lớp Segment (thêm cơ chế steepness) tại: {segment_file}")
    else:
        print(f"[-] Lớp Segment đã được cập nhật từ trước.")

    # =====================================================================
    # 2. CẬP NHẬT TỆP: test_simulation.py
    # =====================================================================
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()

    old_magnet_setup = """    i_magnet_left = Segment(
        outline=[[-0.04, -0.04], [-0.02, -0.04], [-0.02, 0.04], [-0.04, 0.04]],
        material="i_bar_left", 
        relative_permeability=1.05, 
        coercive=[0.0, 800000.0]
    )

    i_magnet_right = Segment(
        outline=[[0.02, -0.04], [0.04, -0.04], [0.04, 0.04], [0.02, 0.04]],
        material="i_bar_right", 
        relative_permeability=1.05, 
        coercive=[0.0, -800000.0]
    )"""

    new_magnet_setup = """    i_magnet_left = Segment(
        outline=[[-0.04, -0.04], [-0.02, -0.04], [-0.02, 0.04], [-0.04, 0.04]],
        material="i_bar_left", 
        relative_permeability=1.05, 
        coercive=[0.0, 800000.0],
        steepness=0.1  # BỔ SUNG: Làm mượt ranh giới
    )

    i_magnet_right = Segment(
        outline=[[0.02, -0.04], [0.04, -0.04], [0.04, 0.04], [0.02, 0.04]],
        material="i_bar_right", 
        relative_permeability=1.05, 
        coercive=[0.0, -800000.0],
        steepness=0.1  # BỔ SUNG: Làm mượt ranh giới
    )"""

    if "steepness=0.1" not in test_content:
        test_content = test_content.replace(old_magnet_setup, new_magnet_setup)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"[*] Đã thiết lập steepness=0.1 cho 2 nam châm trong: {test_file}")
    else:
        print(f"[-] Các nam châm đã được thiết lập steepness từ trước.")
        
    print("\\n[+] ĐÃ HOÀN TẤT CẬP NHẬT STEEPNESS!")

if __name__ == "__main__":
    update_steepness_config()