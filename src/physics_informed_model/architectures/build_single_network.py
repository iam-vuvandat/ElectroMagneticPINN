import torch.nn as nn

def _build_single_network(self, actual_input_dim, network_config, out_dim=None):
    layers = []
    hidden_neurons = network_config.hidden_neurons
    hidden_layers = network_config.hidden_layers
    activation = network_config.activation_function
    target_out_dim = out_dim if out_dim is not None else self.output_dim

    layers.append(nn.Linear(actual_input_dim, hidden_neurons))
    layers.append(activation)

    for _ in range(1, hidden_layers):
        layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        layers.append(activation)

    layers.append(nn.Linear(hidden_neurons, target_out_dim))
    return nn.Sequential(*layers)