import torch

def build_domain(self):
    if self.geometry_config.outline is not None:
        if len(self.geometry_config.outline) < 3:
            raise ValueError(f"Lỗi: Outline của domain '{self.domain_name}' phải có ít nhất 3 đỉnh.")
        self.outline_tensor = torch.tensor(self.geometry_config.outline, dtype=torch.float32)
        self._compute_section_area()
        if self.section_area <= 1e-12:
            raise ValueError(f"Lỗi: Diện tích của domain '{self.domain_name}' bằng 0 hoặc suy biến.")
        self._compute_current_density()
    else:
        raise ValueError(f"Lỗi: Domain '{self.domain_name}' chưa được cấp outline trong geometry_config.")
        
    if self.physics_config.bh_curve_data is None:
        self._compute_constant_bh_curve()
        
    return self
