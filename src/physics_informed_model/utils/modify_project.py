import os

def extract_physics_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    # =====================================================================
    # 1. CẬP NHẬT TỆP: electro_magnetic_pinn.py
    # =====================================================================
    pinn_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    with open(pinn_file, 'r', encoding='utf-8') as f:
        pinn_content = f.read()

    # Thêm physics_config vào hàm khởi tạo
    old_init_block = """        self.visualization_config = SimpleNamespace(
            active=False,
            update_interval=100,
            output_directory="animation_frames",
            resolution=100,
            gif_filename="training_process.gif",
            gif_fps=10
        )
        
        self.geometry_engine_instance = Geometry()"""
        
    new_init_block = """        self.visualization_config = SimpleNamespace(
            active=False,
            update_interval=100,
            output_directory="animation_frames",
            resolution=100,
            gif_filename="training_process.gif",
            gif_fps=10
        )
        
        # BỔ SUNG: Cấu hình chuẩn hóa vật lý
        self.physics_config = SimpleNamespace(
            scale_L0=0.05,
            scale_H0=1200000.0,
            scale_nu0=VACUUM_RELUCTIVITY
        )
        
        self.geometry_engine_instance = Geometry()"""

    # Loại bỏ giá trị hardcode trong _build_system
    old_build_block = """        self.L0 = self.collocation_sampler_instance.x_maximum
        self.H0 = 800000.0
        self.nu0 = VACUUM_RELUCTIVITY"""
        
    new_build_block = """        # SỬ DỤNG GIÁ TRỊ TỪ CONFIG THAY VÌ GÁN CỨNG
        self.L0 = self.physics_config.scale_L0
        self.H0 = self.physics_config.scale_H0
        self.nu0 = self.physics_config.scale_nu0"""

    if "self.physics_config = SimpleNamespace" not in pinn_content:
        pinn_content = pinn_content.replace(old_init_block, new_init_block)
        pinn_content = pinn_content.replace(old_build_block, new_build_block)
        with open(pinn_file, 'w', encoding='utf-8') as f:
            f.write(pinn_content)
        print(f"[*] Đã trích xuất cấu hình vật lý vào: {pinn_file}")

    # =====================================================================
    # 2. CẬP NHẬT TỆP: maxwell_pde_loss.py
    # =====================================================================
    pde_file = os.path.join(project_root, 'physics_domain', 'physical_equations', 'maxwell_pde_loss.py')
    with open(pde_file, 'r', encoding='utf-8') as f:
        pde_content = f.read()

    old_pde_init = "def __init__(self, L0=0.05, H0=800000.0, nu0=795774.715459):"
    new_pde_init = "def __init__(self, L0=0.05, H0=1200000.0, nu0=795774.715459):"

    if old_pde_init in pde_content:
        pde_content = pde_content.replace(old_pde_init, new_pde_init)
        with open(pde_file, 'w', encoding='utf-8') as f:
            f.write(pde_content)
        print(f"[*] Đã sửa giá trị mặc định H0 trong: {pde_file}")

    # =====================================================================
    # 3. CẬP NHẬT TỆP: test_simulation.py
    # =====================================================================
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()

    # Chèn cấu hình thiết lập H0 bằng 1.5 lần max_coercive_field
    old_test_setup = """    model = ElectroMagneticPINN()
    
    # Thiết lập miền không gian tính toán"""
    
    new_test_setup = """    model = ElectroMagneticPINN()
    
    # Thiết lập hệ số chuẩn hóa H0 (1.5 lần lực kháng từ cực đại)
    max_coercive_field = 800000.0
    model.physics_config.scale_H0 = max_coercive_field * 1.5
    
    # Thiết lập miền không gian tính toán"""

    if "model.physics_config.scale_H0" not in test_content:
        test_content = test_content.replace(old_test_setup, new_test_setup)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"[*] Đã thiết lập H0 động (1.5x) vào: {test_file}")

if __name__ == "__main__":
    extract_physics_config()