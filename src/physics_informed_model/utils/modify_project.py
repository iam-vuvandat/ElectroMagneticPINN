import os

def apply_fourier_feature_encoding():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    arch_file = os.path.join(project_root, 'pinn_architecture.py')
    with open(arch_file, 'r', encoding='utf-8') as f:
        arch_content = f.read()

    old_arch_import = """import torch
import torch.nn as nn

class PINNArchitecture(nn.Module):"""

    new_arch_import = """import torch
import torch.nn as nn
import numpy as np

class PINNArchitecture(nn.Module):"""

    old_arch_init = """    def __init__(self, input_dim=2, hidden_layers=4, hidden_neurons=50, output_dim=1, domain_scale=0.05, activation_function=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.domain_scale = domain_scale"""

    new_arch_init = """    def __init__(self, input_dim=2, hidden_layers=4, hidden_neurons=50, output_dim=1, domain_scale=0.05, activation_function=None, fourier_features=False, fourier_sigma=1.0, fourier_mapping_size=256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.domain_scale = domain_scale
        
        self.fourier_features = fourier_features
        self.fourier_sigma = fourier_sigma
        self.fourier_mapping_size = fourier_mapping_size
        
        if self.fourier_features:
            B_matrix = torch.randn(self.fourier_mapping_size, self.input_dim) * self.fourier_sigma
            self.register_buffer('B_matrix', B_matrix)
            actual_input_dim = self.fourier_mapping_size * 2
        else:
            actual_input_dim = self.input_dim"""

    old_arch_layers = """        layers.append(nn.Linear(self.input_dim, self.hidden_neurons))"""
    new_arch_layers = """        layers.append(nn.Linear(actual_input_dim, self.hidden_neurons))"""

    old_arch_forward = """    def forward(self, xy):
        xy_normalized = xy / self.domain_scale
        raw_output = self.network(xy_normalized)
        
        A_z_star = raw_output * self.boundary_factor(xy)
        return A_z_star"""

    new_arch_forward = """    def forward(self, xy):
        xy_normalized = xy / self.domain_scale
        
        if self.fourier_features:
            projected = (2.0 * np.pi * xy_normalized) @ self.B_matrix.T
            x_input = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
        else:
            x_input = xy_normalized
            
        raw_output = self.network(x_input)
        A_z_star = raw_output * self.boundary_factor(xy)
        return A_z_star"""

    if "fourier_features" not in arch_content:
        if "import numpy as np" not in arch_content:
            arch_content = arch_content.replace(old_arch_import, new_arch_import)
        arch_content = arch_content.replace(old_arch_init, new_arch_init)
        arch_content = arch_content.replace(old_arch_layers, new_arch_layers)
        arch_content = arch_content.replace(old_arch_forward, new_arch_forward)
        with open(arch_file, 'w', encoding='utf-8') as f:
            f.write(arch_content)

    pinn_file = os.path.join(project_root, 'electro_magnetic_pinn.py')
    with open(pinn_file, 'r', encoding='utf-8') as f:
        pinn_content = f.read()

    old_pinn_config = """        self.pinn_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU()
        )"""

    new_pinn_config = """        self.pinn_config = SimpleNamespace(
            hidden_layers=4,
            hidden_neurons=64,
            activation_function=nn.SiLU(),
            fourier_features=False,
            fourier_sigma=1.0,
            fourier_mapping_size=256
        )"""

    if "fourier_features" not in pinn_content:
        pinn_content = pinn_content.replace(old_pinn_config, new_pinn_config)
        with open(pinn_file, 'w', encoding='utf-8') as f:
            f.write(pinn_content)

    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()

    old_test_setup = """    model.pinn_config.hidden_layers = 6
    model.pinn_config.hidden_neurons = 64
    model.pinn_config.activation_function = nn.Tanh()
    
    model.training_config.epochs_adam = 4000"""

    new_test_setup = """    model.pinn_config.hidden_layers = 6
    model.pinn_config.hidden_neurons = 64
    model.pinn_config.activation_function = nn.Tanh()
    
    model.pinn_config.fourier_features = True
    model.pinn_config.fourier_sigma = 2.0
    model.pinn_config.fourier_mapping_size = 256
    
    model.training_config.epochs_adam = 4000"""

    if "model.pinn_config.fourier_features" not in test_content:
        test_content = test_content.replace(old_test_setup, new_test_setup)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

if __name__ == "__main__":
    apply_fourier_feature_encoding()