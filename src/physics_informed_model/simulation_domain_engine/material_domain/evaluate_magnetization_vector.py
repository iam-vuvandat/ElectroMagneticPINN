import torch

def evaluate_magnetization_vector(self, points_tensor):
    coercive_cfg = self.physics_config.coercive
    if callable(coercive_cfg):
        res = coercive_cfg(points_tensor)
        return res[:, 0:1], res[:, 1:2]
    if isinstance(coercive_cfg, torch.Tensor):
        c_tensor = coercive_cfg.to(device=points_tensor.device, dtype=points_tensor.dtype)
        if c_tensor.dim() == 1:
            hx = torch.full((points_tensor.shape[0], 1), c_tensor[0].item(), dtype=points_tensor.dtype, device=points_tensor.device)
            hy = torch.full((points_tensor.shape[0], 1), c_tensor[1].item(), dtype=points_tensor.dtype, device=points_tensor.device)
            return hx, hy
        return c_tensor[:, 0:1], c_tensor[:, 1:2]
    hx_tensor = torch.full((points_tensor.shape[0], 1), float(coercive_cfg[0]), dtype=points_tensor.dtype, device=points_tensor.device)
    hy_tensor = torch.full((points_tensor.shape[0], 1), float(coercive_cfg[1]), dtype=points_tensor.dtype, device=points_tensor.device)
    return hx_tensor, hy_tensor
