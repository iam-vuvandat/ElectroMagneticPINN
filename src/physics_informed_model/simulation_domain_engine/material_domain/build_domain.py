# src/physics_informed_model/simulation_domain_engine/material_domain/build_domain.py
import torch

def build_domain(self):
    if self.geometry_config.outline is not None:
        if len(self.geometry_config.outline) < 3:
            raise ValueError(f"Error: Outline of domain '{self.domain_name}' must have at least 3 vertices.")
        
        self.outline_tensor = torch.tensor(self.geometry_config.outline, dtype=torch.float32)
        self._compute_section_area()
        if self.section_area <= 1e-12:
            raise ValueError(f"Error: Area of domain '{self.domain_name}' is zero or degenerate.")
        self._compute_current_density()

        starts = self.outline_tensor
        ends = torch.roll(self.outline_tensor, shifts=-1, dims=0)
        edge_vecs = ends - starts
        edge_lens = torch.sqrt(torch.sum(edge_vecs ** 2, dim=1, keepdim=True) + 1e-12)

        signed_area = 0.5 * torch.sum(starts[:, 0] * ends[:, 1] - ends[:, 0] * starts[:, 1])
        orient = 1.0 if signed_area.item() >= 0.0 else -1.0

        unit_tangents = orient * (edge_vecs / edge_lens)
        unit_normals = torch.cat([unit_tangents[:, 1:2], -unit_tangents[:, 0:1]], dim=1)

        self.precomputed_edge_starts = starts
        self.precomputed_edge_vecs = edge_vecs
        self.precomputed_edge_lens = edge_lens
        self.precomputed_unit_tangents = unit_tangents
        self.precomputed_unit_normals = unit_normals

        coercive_cfg = self.physics_config.coercive
        if isinstance(coercive_cfg, (list, tuple)) and len(coercive_cfg) == 2:
            self.coercive_magnitude = float((coercive_cfg[0]**2 + coercive_cfg[1]**2) ** 0.5)
    else:
        raise ValueError(f"Error: Domain '{self.domain_name}' is missing outline in geometry_config.")
        
    if self.physics_config.bh_curve_data is None:
        self._compute_constant_bh_curve()

    bh_data = getattr(self.physics_config, 'bh_curve_data', None)
    if isinstance(bh_data, (list, tuple)) and len(bh_data) == 2:
        self.bh_b_tensor = torch.tensor(bh_data[0], dtype=torch.float32)
        self.bh_h_tensor = torch.tensor(bh_data[1], dtype=torch.float32)
    else:
        self.bh_b_tensor = None
        self.bh_h_tensor = None
        
    return self