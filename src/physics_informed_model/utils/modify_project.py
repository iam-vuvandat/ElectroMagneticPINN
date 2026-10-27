import os

def apply_three_stage_stable_test_update():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_directory, '..'))

    test_file = os.path.join(project_root, 'tests', 'test_simulation.py')

    files_to_update = {}

    # =========================================================================
    # CẬP NHẬT: tests/test_simulation.py
    # (Kích hoạt đủ 3 giai đoạn huấn luyện + Trọng số loss phi tuyến nhỏ = 0.01)
    # =========================================================================
    files_to_update[test_file] = '''import os
import sys
import shutil
from types import SimpleNamespace

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from electro_magnetic_pinn import ElectroMagneticPINN

def run_modern_simulation_test():
    output_dir = "figure_modern_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("[*] Bắt đầu kiểm thử XPINN với đầy đủ 3 Giai đoạn Huấn luyện (Warm-up, Adaptation, L-BFGS)...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # 1. Cấu hình chuẩn hóa trường vật lý
    sim_domain.physics_config.scale_H0 = 800000.0 * 1.2
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.x_boundaries_tuple = (-0.1, 0.1)
    sim_domain.y_boundaries_tuple = (-0.1, 0.1)

    # 2. Cấu hình Lấy mẫu động
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # 3. KÍCH HOẠT ĐỦ 3 GIAI ĐOẠN HUẤN LUYỆN
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = True

    # Phân bổ Epochs: 1000 Adam Warm-up (40%) + 1500 Adam Non-linear Adaptation (60%) + 600 L-BFGS
    model.training_config.number_of_adam_epochs = 2500
    model.training_config.linear_warmup_epochs_ratio = 0.4
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-7

    model.training_config.number_of_lbfgs_iterations = 600
    model.training_config.lbfgs_learning_rate = 0.5
    model.training_config.lbfgs_maximum_evaluations = 800

    # 4. Cấu hình Trực quan hóa
    model.visualization_config.active = True
    model.visualization_config.update_interval = 200
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 80
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "three_stage_training.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "three_stage_training.mp4"

    # Đặt trọng số pde_nonlin_w ở mức rất nhỏ (mặc định = 0.01) để tăng độ ổn định gradient
    def apply_standard_config(domain, int_pts=2000, surf_pts=1200, pde_nonlin_w=0.01):
        domain.training_config.pde_weight = 1.0
        domain.training_config.pde_nonlinear_weight = pde_nonlin_w  # Trọng số phi tuyến nhỏ dập bùng nổ gradient
        domain.training_config.interface_weight = 1.5
        domain.training_config.interface_potential_weight = 1000.0
        domain.training_config.interface_bn_weight = 400.0
        domain.training_config.interface_ht_weight = 400.0
        domain.training_config.pde_fading_beta = 80.0
        domain.training_config.number_of_interior_points = int_pts
        domain.training_config.number_of_surface_points = surf_pts
        
        domain.network_config.hidden_layers = 4
        domain.network_config.hidden_neurons = 64
        domain.network_config.activation_function = nn.SiLU()
        domain.network_config.fourier_features = True
        domain.network_config.fourier_sigma = 1.2
        domain.network_config.fourier_mapping_size = 128
        return domain.build_domain()

    # 1. Khung Ách sắt (Iron Yoke - steel_1008 phi tuyến)
    iron_yoke = sim_domain.create_material_domain(
        domain_name="iron_yoke",
        material_name="steel_1008",
        outline=[
            [-0.06, -0.03], 
            [0.06, -0.03], 
            [0.06, 0.0], 
            [-0.06, 0.0]
        ]
    )
    apply_standard_config(iron_yoke, int_pts=3000, surf_pts=1600, pde_nonlin_w=0.01)

    # 2. Nam châm trái (N30UH)
    magnet_left = sim_domain.create_material_domain(
        domain_name="magnet_left",
        material_name="N30UH",
        outline=[
            [-0.05, 0.0], 
            [-0.005, 0.0], 
            [-0.005, 0.018], 
            [-0.05, 0.018]
        ],
        coercive_direction=(0, 1)
    )
    apply_standard_config(magnet_left, int_pts=2000, surf_pts=1200, pde_nonlin_w=0.01)

    # 3. Nam châm phải (N30UH)
    magnet_right = sim_domain.create_material_domain(
        domain_name="magnet_right",
        material_name="N30UH",
        outline=[
            [0.005, 0.0], 
            [0.05, 0.0], 
            [0.05, 0.018], 
            [0.005, 0.018]
        ],
        coercive_direction=(0, -1)
    )
    apply_standard_config(magnet_right, int_pts=2000, surf_pts=1200, pde_nonlin_w=0.01)

    # 4. Background Air
    bg_training_cfg = SimpleNamespace(
        pde_weight=1.0,
        pde_nonlinear_weight=0.01,
        interface_weight=1.5,
        pde_fading_beta=80.0,
        sampling_density_multiplier=1.0,
        number_of_interior_points=8000,
        number_of_surface_points=0
    )
    bg_network_cfg = SimpleNamespace(
        hidden_layers=4,
        hidden_neurons=64,
        activation_function=nn.SiLU(),
        fourier_features=True,
        fourier_sigma=1.2,
        fourier_mapping_size=128
    )
    sim_domain.build_background_domain(
        domain_name="background",
        training_config=bg_training_cfg,
        network_config=bg_network_cfg
    )

    model.update_electromagnetic_pinn()
    sim_domain.plot_problem_definition(resolution=100)
    
    print("[*] Chạy tiến trình huấn luyện 3 giai đoạn...")
    model.execute_training_process()

    print("[*] Đang xuất biểu đồ kết quả trường từ sau khi hoàn thành L-BFGS...")
    model.plot_fields(
        resolution=180, 
        save_path=f"{output_dir}/three_stage_final_fields.png"
    )

    print(f"[✔] HOÀN THÀNH KIỂM THỬ 3 GIAI ĐOẠN! Kết quả lưu tại {output_dir}/")

def test_simulation4():
    run_modern_simulation_test()

if __name__ == "__main__":
    run_modern_simulation_test()
'''

    for file_path, code_content in files_to_update.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        print(f"[*] Đã ghi và cập nhật thành công: {os.path.relpath(file_path, project_root)}")

    print("\n[✔] HOÀN THÀNH CẬP NHẬT FILE TEST VỚI 3 GIAI ĐOẠN HUẤN LUYỆN VÀ LOSS PHÍ TUYẾN ỔN ĐỊNH!")

if __name__ == "__main__":
    apply_three_stage_stable_test_update()