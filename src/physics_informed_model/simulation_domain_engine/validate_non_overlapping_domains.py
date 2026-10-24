import torch

def validate_non_overlapping_domains(self, candidate_domain=None, check_resolution=80):
    if getattr(self.physics_config, 'allow_overlapping_domains', False):
        return True
    mat_domains = [d for d in self.domains_list if getattr(d, 'domain_name', '') != 'background']
    if candidate_domain is not None and getattr(candidate_domain, 'domain_name', '') != 'background':
        if candidate_domain not in mat_domains:
            mat_domains.append(candidate_domain)
    if len(mat_domains) < 2:
        return True

    x_min, x_max = self.x_boundaries_tuple
    y_min, y_max = self.y_boundaries_tuple
    xs = torch.linspace(x_min, x_max, check_resolution)
    ys = torch.linspace(y_min, y_max, check_resolution)
    Xg, Yg = torch.meshgrid(xs, ys, indexing='ij')
    test_pts = torch.column_stack([Xg.reshape(-1), Yg.reshape(-1)])

    strict_inside_counts = torch.zeros(test_pts.shape[0], dtype=torch.int32)
    for d in mat_domains:
        sdf = d.compute_signed_distance_field(test_pts).view(-1)
        strict_inside_counts += (sdf < -1e-6).int()

    if torch.any(strict_inside_counts > 1):
        raise ValueError("Lỗi hình học: Phát hiện các miền vật liệu chồng lấn lên nhau.")
    return True
