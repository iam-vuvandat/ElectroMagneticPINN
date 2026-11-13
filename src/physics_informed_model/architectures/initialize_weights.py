import torch.nn as nn

def _initialize_weights(self):
    for network_key, network in self.networks.items():
        for module in network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
