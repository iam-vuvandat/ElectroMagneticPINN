import torch
import torch.nn as nn

class PINNArchitecture(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=4, hidden_neurons=50, output_dim=1, domain_scale=0.05, activation_function=nn.SiLU()):
        super().__init__()
        self.domain_scale = domain_scale
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(activation_function)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(activation_function)
            
        layers.append(nn.Linear(hidden_neurons, output_dim))
        
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
        raw_output = self.network(xy_normalized)
        
        A_z_star = raw_output * self.boundary_factor(xy)
        return A_z_star
