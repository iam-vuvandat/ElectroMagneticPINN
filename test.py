import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
physics_informed_model_path = os.path.join(current_directory, 'src', 'physics_informed_model')
if physics_informed_model_path not in sys.path:
    sys.path.insert(0, physics_informed_model_path)

import torch
from simulation_domain_engine.simulation_domain import SimulationDomain
from simulation_domain_engine.material_domain.material_domain import MaterialDomain

def run_simulation_domain_engine_test():
    sim_domain = SimulationDomain(
        x_boundaries_tuple=(-0.06, 0.06),
        y_boundaries_tuple=(-0.06, 0.06)
    )

    magnet_left = MaterialDomain(domain_name="magnet_left")
    magnet_left.geometry_config.outline = [
        [-0.04, -0.03],
        [-0.02, -0.03],
        [-0.02, 0.03],
        [-0.04, 0.03]
    ]
    magnet_left.physics_config.relative_permeability = 1.05
    magnet_left.physics_config.coercive = [0.0, 800000.0]
    magnet_left.training_config.pde_weight = 2.0
    magnet_left.build_domain()

    magnet_right = MaterialDomain(domain_name="magnet_right")
    magnet_right.geometry_config.outline = [
        [0.02, -0.03],
        [0.04, -0.03],
        [0.04, 0.03],
        [0.02, 0.03]
    ]
    magnet_right.physics_config.relative_permeability = 1.05
    magnet_right.physics_config.coercive = [0.0, -800000.0]
    magnet_right.training_config.pde_weight = 2.0
    magnet_right.build_domain()

    sim_domain.add_material_domain(magnet_left)
    sim_domain.add_material_domain(magnet_right)

    sim_domain.build_background_domain(domain_name="background")

    domain_count = sim_domain.get_domain_count()
    assert domain_count == 3, f"Lỗi số lượng domain: kỳ vọng 3, nhận được {domain_count}"
    assert sim_domain.domains_list[-1] == sim_domain.background_domain, "Lỗi: background_domain không nằm ở cuối domains_list"

    assert magnet_left.domain_id == 1, f"Lỗi domain_id magnet_left: {magnet_left.domain_id}"
    assert magnet_right.domain_id == 2, f"Lỗi domain_id magnet_right: {magnet_right.domain_id}"
    assert sim_domain.background_domain.domain_id == 3, f"Lỗi domain_id background: {sim_domain.background_domain.domain_id}"

    total_box_area = 0.12 * 0.12
    sub_materials_area = magnet_left.section_area + magnet_right.section_area
    expected_bg_area = total_box_area - sub_materials_area
    assert abs(sim_domain.background_domain.section_area - expected_bg_area) < 1e-7, "Lỗi tính toán diện tích background"

    test_points = torch.tensor([
        [-0.03, 0.0],
        [0.03, 0.0],
        [0.0, 0.0]
    ], dtype=torch.float32)

    sdf_values = sim_domain.compute_global_signed_distance_field(test_points).squeeze()
    assert sdf_values[0] <= 0.0, "Lỗi SDF: Điểm trong magnet_left phải có SDF <= 0"
    assert sdf_values[1] <= 0.0, "Lỗi SDF: Điểm trong magnet_right phải có SDF <= 0"
    assert sdf_values[2] > 0.0, "Lỗi SDF: Điểm trong background phải có SDF > 0"

    props = sim_domain.evaluate_global_physical_properties(test_points)

    assert props["material_classification"][0, 0].item() == 1.0, "Lỗi ID vật liệu tại magnet_left"
    assert props["coercive_field_y"][0, 0].item() == 800000.0, "Lỗi Hc_y tại magnet_left"

    assert props["material_classification"][1, 0].item() == 2.0, "Lỗi ID vật liệu tại magnet_right"
    assert props["coercive_field_y"][1, 0].item() == -800000.0, "Lỗi Hc_y tại magnet_right"

    assert props["material_classification"][2, 0].item() == 3.0, "Lỗi ID vật liệu tại background"
    assert props["coercive_field_y"][2, 0].item() == 0.0, "Lỗi Hc_y tại background"
    assert abs(props["reluctivity"][2, 0].item() - 795774.715459) < 1.0, "Lỗi độ từ trở background"

    sim_domain.display(resolution=100)

if __name__ == "__main__":
    run_simulation_domain_engine_test()