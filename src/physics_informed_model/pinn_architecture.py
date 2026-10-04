import torch
import torch.nn as nn

class PINNArchitecture(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=4, hidden_neurons=50, output_dim=1, domain_scale=0.05, activation_function=None):
        super().__init__()
        # ĐƯA RA LÀM THUỘC TÍNH
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.domain_scale = domain_scale
        
        if activation_function is None:
            self.activations = [nn.SiLU() for _ in range(self.hidden_layers)]
        elif isinstance(activation_function, list):
            self.activations = activation_function
        else:
            self.activations = [activation_function for _ in range(self.hidden_layers)]
            
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_neurons))
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
        raw_output = self.network(xy_normalized)
        return raw_output * self.boundary_factor(xy)
