import torch
from architectures.xpinn_architecture import XPINNArchitecture
from training_engine.training_manager import TrainingManager
from physics_domain.physical_equations.maxwell_pde_loss import MaxwellPDELoss
from physics_domain.physical_equations.interface_loss import InterfaceLoss
from physics_domain.collocation_sampler import CollocationSampler
from utils.training_visualizer import TrainingVisualizer

def _build_system(pinn_instance):
    pinn_instance.simulation_domain_engine_instance.validate_non_overlapping_domains()
    pinn_instance.computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hệ thống vật lý được khởi tạo trên: {pinn_instance.computation_device}")
    
    pinn_instance.collocation_sampler_instance = CollocationSampler(
        electro_magnetic_pinn=pinn_instance,
        device=pinn_instance.computation_device
    )
    
    pinn_instance.pinn_architecture_instance = XPINNArchitecture(
        electro_magnetic_pinn=pinn_instance
    ).to(pinn_instance.computation_device)
    
    pinn_instance.maxwell_pde_loss_instance = MaxwellPDELoss(electro_magnetic_pinn=pinn_instance)
    pinn_instance.interface_loss_instance = InterfaceLoss(electro_magnetic_pinn=pinn_instance)
    
    pinn_instance.visualizer_instance = None
    if pinn_instance.visualization_config.active:
        pinn_instance.visualizer_instance = TrainingVisualizer(electro_magnetic_pinn=pinn_instance)
        
    pinn_instance.training_manager_instance = TrainingManager(electro_magnetic_pinn=pinn_instance)

    domains_data = pinn_instance.collocation_sampler_instance.generate_all_domain_points()
    pre_ep = getattr(pinn_instance.training_config, 'pretrain_nu_epochs', 300)
    pre_lr = getattr(pinn_instance.training_config, 'pretrain_nu_lr', 1e-2)
    pinn_instance.training_manager_instance.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)
