# src/physics_informed_model/simulation_domain_engine/material_domain/evaluate_reluctivity.py
import torch
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

def evaluate_reluctivity(self, points_tensor, B_magnitude_tensor=None, is_nonlinear=None):
    if is_nonlinear is None:
        is_nonlinear = getattr(self.physics_config, 'is_nonlinear', False)

    if not is_nonlinear:
        mu_r = float(getattr(self.physics_config, 'relative_permeability', 1.0))
        const_nu = VACUUM_RELUCTIVITY / max(mu_r, 1e-6)
        return torch.full(
            (points_tensor.shape[0], 1),
            const_nu,
            dtype=points_tensor.dtype,
            device=points_tensor.device
        )

    b_pts = getattr(self, 'bh_b_tensor', None)
    h_pts = getattr(self, 'bh_h_tensor', None)

    if b_pts is None or h_pts is None:
        bh_data = getattr(self.physics_config, 'bh_curve_data', None)
        if isinstance(bh_data, (list, tuple)) and len(bh_data) == 2:
            b_pts = torch.tensor(bh_data[0], dtype=points_tensor.dtype, device=points_tensor.device)
            h_pts = torch.tensor(bh_data[1], dtype=points_tensor.dtype, device=points_tensor.device)
            self.bh_b_tensor = b_pts
            self.bh_h_tensor = h_pts
        else:
            raise ValueError(f"Domain '{self.domain_name}' is missing valid bh_curve_data.")

    if b_pts.device != points_tensor.device or b_pts.dtype != points_tensor.dtype:
        b_pts = b_pts.to(device=points_tensor.device, dtype=points_tensor.dtype)
        h_pts = h_pts.to(device=points_tensor.device, dtype=points_tensor.dtype)
        self.bh_b_tensor = b_pts
        self.bh_h_tensor = h_pts

    if B_magnitude_tensor is None:
        B_magnitude_tensor = torch.zeros(
            (points_tensor.shape[0], 1),
            dtype=points_tensor.dtype,
            device=points_tensor.device
        )

    b_flat = B_magnitude_tensor.view(-1)
    indices = torch.bucketize(b_flat, b_pts, right=True) - 1
    indices = torch.clamp(indices, 0, len(b_pts) - 2)

    b0, b1 = b_pts[indices], b_pts[indices + 1]
    h0, h1 = h_pts[indices], h_pts[indices + 1]

    t = (b_flat - b0) / (b1 - b0 + 1e-12)
    h_interp = h0 + t * (h1 - h0)

    nu_zero = h_pts[1] / torch.clamp(b_pts[1], min=1e-12)
    nu_tensor = torch.where(b_flat > 1e-7, h_interp / torch.clamp(b_flat, min=1e-7), nu_zero)
    return nu_tensor.view(-1, 1)