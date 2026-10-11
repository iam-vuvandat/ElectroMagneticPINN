import torch
import torch.nn as nn
import numpy as np

class PINNArchitecture(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=4, hidden_neurons=50, output_dim=1, domain_scale=0.05, activation_function=None, fourier_features=False, fourier_sigma=1.0, fourier_mapping_size=256):
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
            actual_input_dim = self.input_dim
        
        
        if activation_function is None:
            
            self.activations = [nn.SiLU() for _ in range(self.hidden_layers)]
        elif isinstance(activation_function, list):
            
            if len(activation_function) != self.hidden_layers:
                raise ValueError(f"Lỗi: Có {self.hidden_layers} lớp ẩn, nhưng nhận được {len(activation_function)} hàm kích hoạt.")
            self.activations = activation_function
        else:
            
            self.activations = [activation_function for _ in range(self.hidden_layers)]
            
      
        layers = []
        
        
        layers.append(nn.Linear(actual_input_dim, self.hidden_neurons))
        layers.append(self.activations[0])
        
        
        for i in range(1, self.hidden_layers):
            layers.append(nn.Linear(self.hidden_neurons, self.hidden_neurons))
            layers.append(self.activations[i]) 
            
        
        layers.append(nn.Linear(self.hidden_neurons, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def boundary_factor(self, xy):
        x_factor = 1.0 - (xy[:, 0:1] / self.domain_scale)**2
        y_factor = 1.0 - (xy[:, 1:2] / self.domain_scale)**2
        return x_factor * y_factor

    def forward(self, xy):
        xy_normalized = xy / self.domain_scale
        
        if self.fourier_features:
            projected = (2.0 * np.pi * xy_normalized) @ self.B_matrix.T
            x_input = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
        else:
            x_input = xy_normalized
            
        raw_output = self.network(x_input)
        A_z_star = raw_output * self.boundary_factor(xy)
        return A_z_star