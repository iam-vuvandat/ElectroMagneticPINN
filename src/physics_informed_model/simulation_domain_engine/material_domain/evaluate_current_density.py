import torch

def evaluate_current_density(self, points_tensor):
    if callable(self.current_density):
        val = self.current_density(points_tensor)
        return val.view(-1, 1).to(device=points_tensor.device, dtype=points_tensor.dtype)
    return torch.full((points_tensor.shape[0], 1), float(self.current_density), dtype=points_tensor.dtype, device=points_tensor.device)
