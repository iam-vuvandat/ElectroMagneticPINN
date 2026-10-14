import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
from electro_magnetic_pinn import ElectroMagneticPINN
from simulation_domain_engine.material_domain.material_domain import MaterialDomain

def run_geometry_and_physics_tests():
    model = ElectroMagneticPINN()
    sim = model.simulation_domain_engine_instance
    sim.x_boundaries_tuple = (-0.1, 0.1)
    sim.y_boundaries_tuple = (-0.1, 0.1)
    sim.physics_config.scale_L0 = 0.1
    sim.physics_config.scale_H0 = 800000.0

    mag_left = MaterialDomain(domain_name="magnet_left")
    mag_left.geometry_config.outline = [
        [-0.04, -0.04],
        [-0.02, -0.04],
        [-0.02, 0.04],
        [-0.04, 0.04]
    ]
    mag_left.physics_config.relative_permeability = 1.05
    mag_left.physics_config.coercive = [0.0, 800000.0]
    mag_left.training_config.number_of_interior_points = 400
    mag_left.training_config.number_of_surface_points = 100
    mag_left.training_config.sampling_density_multiplier = 1.5
    mag_left.build_domain()

    mag_right = MaterialDomain(domain_name="magnet_right")
    mag_right.geometry_config.outline = [
        [0.02, -0.04],
        [0.04, -0.04],
        [0.04, 0.04],
        [0.02, 0.04]
    ]
    mag_right.physics_config.relative_permeability = 1.05
    mag_right.physics_config.coercive = [0.0, -800000.0]
    mag_right.training_config.number_of_interior_points = 400
    mag_right.training_config.number_of_surface_points = 100
    mag_right.training_config.sampling_density_multiplier = 0.5
    mag_right.build_domain()

    sim.add_material_domain(mag_left)
    sim.add_material_domain(mag_right)
    bg = sim.build_background_domain(domain_name="background")
    bg.training_config.number_of_interior_points = 1000
    bg.training_config.sampling_density_multiplier = 1.0

    model.update_electromagnetic_pinn()

    test_pts = torch.tensor([
        [-0.03, 0.0],
        [0.03, 0.0],
        [0.0, 0.0],
        [-0.1, 0.0],
        [0.1, 0.05]
    ], dtype=torch.float32, device=model.computation_device)

    props = sim.evaluate_global_physical_properties(test_pts)
    mat_ids = props["material_classification"].squeeze().cpu().tolist()
    assert mat_ids[0] == 1.0, f"Điểm trong nam châm trái phải có ID 1.0, nhận được {mat_ids[0]}"
    assert mat_ids[1] == 2.0, f"Điểm trong nam châm phải phải có ID 2.0, nhận được {mat_ids[1]}"
    assert mat_ids[2] == 3.0, f"Điểm giữa khe hở không khí phải có ID 3.0, nhận được {mat_ids[2]}"

    boundary_pts = torch.tensor([
        [-0.1, 0.0],
        [0.1, 0.0],
        [0.0, -0.1],
        [0.0, 0.1]
    ], dtype=torch.float32, device=model.computation_device)
    Az_bound = model.pinn_architecture_instance.forward_network('background', boundary_pts)
    assert torch.allclose(Az_bound, torch.zeros_like(Az_bound), atol=1e-6), "Az tại biên ngoài Dirichlet phải bằng 0."

    domains_data = model.collocation_sampler_instance.generate_all_domain_points()
    assert domains_data['domain_1']['interior_points'].shape[0] == 600, "Sai số điểm khi nhân multiplier = 1.5"
    assert domains_data['domain_1']['surface_points'].shape[0] == 150, "Sai số điểm bề mặt khi nhân multiplier = 1.5"
    assert domains_data['domain_2']['interior_points'].shape[0] == 200, "Sai số điểm khi nhân multiplier = 0.5"
    assert domains_data['domain_2']['surface_points'].shape[0] == 50, "Sai số điểm bề mặt khi nhân multiplier = 0.5"

    loss_val = model.training_manager_instance.compute_loss(domains_data)
    assert torch.isfinite(loss_val), "Tổng loss khởi tạo phải hữu hạn."

    Az_f, Bx_f, By_f = model.evaluate_fields(test_pts)
    assert torch.all(torch.isfinite(Az_f)) and torch.all(torch.isfinite(Bx_f)) and torch.all(torch.isfinite(By_f)), "Trường dự đoán không hữu hạn."

    print("[*] TẤT CẢ CÁC BÀI KIỂM THỬ HÌNH HỌC, LẤY MẪU, BIÊN VÀ VẬT LÝ ĐỀU ĐẠT (PASSED)!")

if __name__ == "__main__":
    run_geometry_and_physics_tests()
