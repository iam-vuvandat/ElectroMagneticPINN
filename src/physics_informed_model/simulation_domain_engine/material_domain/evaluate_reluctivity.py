import torch

def evaluate_reluctivity(self, points_tensor, B_magnitude_tensor=None):
    # 1. Nếu chưa có dữ liệu B-H, tự sinh danh sách B-H tuyến tính từ mu_r
    if self.physics_config.bh_curve_data is None:
        self._compute_constant_bh_curve()

    bh_data = self.physics_config.bh_curve_data

    # 2. Xử lý định dạng list/tuple [B_points, H_points]
    if isinstance(bh_data, (list, tuple)) and len(bh_data) == 2:
        b_pts = torch.tensor(bh_data[0], dtype=points_tensor.dtype, device=points_tensor.device)
        h_pts = torch.tensor(bh_data[1], dtype=points_tensor.dtype, device=points_tensor.device)

        if B_magnitude_tensor is None:
            B_magnitude_tensor = torch.zeros((points_tensor.shape[0], 1), dtype=points_tensor.dtype, device=points_tensor.device)

        # Nội suy tuyến tính H(B) từ bảng dữ liệu B-H
        b_flat = B_magnitude_tensor.view(-1)
        indices = torch.bucketize(b_flat, b_pts) - 1
        indices = torch.clamp(indices, 0, len(b_pts) - 2)

        b0, b1 = b_pts[indices], b_pts[indices + 1]
        h0, h1 = h_pts[indices], h_pts[indices + 1]

        t = (b_flat - b0) / (b1 - b0 + 1e-12)
        h_interp = h0 + t * (h1 - h0)

        # nu(B) = H / B (tại B = 0 lấy độ nghiêng slope của điểm đầu tiên)
        nu_tensor = torch.where(b_flat > 1e-6, h_interp / b_flat, h_pts[1] / b_pts[1])
        return nu_tensor.view(-1, 1)

    raise TypeError(f"Dữ liệu bh_curve_data của domain '{self.domain_name}' phải là 'list' hoặc 'tuple' dạng [B_points, H_points].")