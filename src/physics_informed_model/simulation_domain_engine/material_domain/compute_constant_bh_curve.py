import torch

VACUUM_RELUCTIVITY = 795774.715459

def compute_constant_bh_curve(self):
    def _default_reluctivity_fn(points_tensor, B_magnitude_tensor=None):
        const_nu = VACUUM_RELUCTIVITY / float(self.physics_config.relative_permeability)
        return torch.full((points_tensor.shape[0], 1), const_nu, dtype=points_tensor.dtype, device=points_tensor.device)
    self.physics_config.bh_curve_data = _default_reluctivity_fn
    return self.physics_config.bh_curve_data
