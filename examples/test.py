import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.physics_informed_model.geometry_engine.geometry import Polygon
from src.physics_informed_model.physics_domain.physical_equations.maxwell_partial_differential_equation_loss import MaxwellPDELoss
from src.physics_informed_model.physics_domain.physical_equations.subdomain_material_mapping import MaterialMapping
from src.physics_informed_model.physics_domain.collocation_sampling import CollocationSampler
from src.physics_informed_model.pinn_architecture import PINNArchitecture
from src.physics_informed_model.training_manager import TrainingManager
from src.physics_informed_model.curriculum_training_manager import CurriculumTrainingManager

def run_training_demo():
    stator_outer = Polygon().set_material("iron").set_vertices([[-2.0, 2.0], [2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])
    stator_inner = Polygon().set_material("air").set_vertices([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
    stator_core = stator_outer - stator_inner
    
    magnet = Polygon().set_material("magnet").set_vertices([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])

    sampler = CollocationSampler(x_bounds=[-2.5, 2.5], y_bounds=[-2.5, 2.5])
    xy_train = sampler.generate_combined_points(geometry=stator_core, num_uniform=1000, num_interface=200, epsilon=0.05)
    
    sdf_outer = stator_outer.compute_sdf(xy_train).unsqueeze(1)
    
    sdf_iron = stator_core.compute_sdf(xy_train)
    iron_mask = (sdf_iron <= 0).to(torch.float32).unsqueeze(1)
    
    sdf_magnet = magnet.compute_sdf(xy_train)
    magnet_mask = (sdf_magnet <= 0).to(torch.float32).unsqueeze(1)
    
    slot_masks_dict = {}
    slot_current_dict = {}
    
    H_cx = torch.zeros_like(iron_mask)
    H_cy = magnet_mask * 800000.0

    model = PINNArchitecture(input_dim=2, hidden_layers=4, hidden_neurons=50, output_dim=1)
    pde_evaluator = MaxwellPDELoss()
    material_mapping = MaterialMapping()
    
    training_manager = TrainingManager(model, pde_evaluator, material_mapping, lr_adam=1e-3)
    curriculum_manager = CurriculumTrainingManager(training_manager)
    
    curriculum_manager.train_source_ramping(
        stages=3,
        epochs_per_stage=100,
        xy=xy_train,
        sdf_boundary=sdf_outer,
        iron_mask=iron_mask,
        magnet_mask=magnet_mask,
        slot_masks_dict=slot_masks_dict,
        slot_current_dict=slot_current_dict,
        H_cx=H_cx,
        H_cy=H_cy
    )

if __name__ == "__main__":
    run_training_demo()