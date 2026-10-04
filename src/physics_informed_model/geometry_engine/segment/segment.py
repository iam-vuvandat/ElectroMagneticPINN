import torch
from geometry_engine.segment.polygon_signed_distance_field import compute_polygon_signed_distance_field

class Segment:
    def __init__(self, 
                 outline=None,
                 material="air",
                 relative_permeability=1.0,
                 bh_curve=None,
                 coercive=[0.0, 0.0],
                 current=0.0):
        
        self.vacuum_reluctivity = 795774.715459
        
        # 1. Gán các thuộc tính cơ bản từ tham số người dùng
        self.material = material
        self.relative_permeability = relative_permeability
        self.coercive = coercive
        self.current = current
        self.bh_curve = bh_curve
        
        # 2. Khởi tạo các thuộc tính nội tại (sẽ tự động tính toán)
        self.outline = None
        self.outline_tensor = None
        self.section_area = 0.0
        self.current_density = 0.0
        self.steepness = 1.0  # Mặc định an toàn
        
        # ==========================================
        # TỰ ĐỘNG CHẠY CÁC PHƯƠNG THỨC CẦN THIẾT
        # ==========================================
        
        # Xử lý hình học và các thông số phụ thuộc (diện tích, dòng điện, độ dốc ranh giới)
        if outline is not None:
            self.set_outline(outline) # Hàm này đã bao gồm compute_section_area()
            self.compute_current_density()
            self.calculate_penetrating_steepness()
            
        # Tự động tạo hàm nội suy độ từ trở nếu không cung cấp bh_curve (vật liệu tuyến tính)
        if self.bh_curve is None:
            self.compute_constant_bh_curve()

    def compute_constant_bh_curve(self):
        constant_reluctivity = self.vacuum_reluctivity / self.relative_permeability
        self.bh_curve = lambda points_tensor: torch.full((points_tensor.shape[0], 1), constant_reluctivity, dtype=torch.float32, device=points_tensor.device)
        return self.bh_curve

    def compute_section_area(self):
        if self.outline is None or len(self.outline) < 3:
            self.section_area = 0.0
            return self.section_area
        x = [p[0] for p in self.outline]
        y = [p[1] for p in self.outline]
        self.section_area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
        return self.section_area

    def compute_current_density(self):
        area = self.compute_section_area() # Đảm bảo area luôn được cập nhật mới nhất
        if area > 0.0:
            self.current_density = self.current / area
        else:
            self.current_density = 0.0
        return self.current_density

    def calculate_penetrating_steepness(self, grid_resolution=50):
        if self.outline_tensor is None or self.outline_tensor.shape[0] < 3:
            self.steepness = 1.0
            return self.steepness
            
        x_min = torch.min(self.outline_tensor[:, 0]).item()
        x_max = torch.max(self.outline_tensor[:, 0]).item()
        y_min = torch.min(self.outline_tensor[:, 1]).item()
        y_max = torch.max(self.outline_tensor[:, 1]).item()
        
        x_coords = torch.linspace(x_min, x_max, grid_resolution)
        y_coords = torch.linspace(y_min, y_max, grid_resolution)
        X_grid, Y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        grid_points = torch.stack([X_grid.ravel(), Y_grid.ravel()], dim=1).to(self.outline_tensor.device)
        
        sdf_values = self.compute_signed_distance_field(grid_points)
        min_sdf = torch.min(sdf_values).item() 
        
        if min_sdf >= 0.0:
            self.steepness = 0.0 
            return self.steepness
            
        d_max = abs(min_sdf)
        
        ideal_k = 6.0 / d_max
        s = (5000.0 - ideal_k) / 4960.0
        
        self.steepness = max(0.0, min(1.0, float(s)))
        return self.steepness

    def set_outline(self, outline):
        self.outline = outline
        self.outline_tensor = torch.tensor(outline, dtype=torch.float32)
        self.compute_section_area()
        return self

    def compute_signed_distance_field(self, points_tensor):
        return compute_polygon_signed_distance_field(self.outline_tensor, points_tensor)

    def evaluate_reluctivity(self, points_tensor):
        # Hàm bh_curve đã được đảm bảo luôn tồn tại nhờ quá trình tự động ở __init__
        return self.bh_curve(points_tensor)

    def evaluate_magnetization_vector(self, points_tensor):
        hx_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[0], dtype=torch.float32, device=points_tensor.device)
        hy_tensor = torch.full((points_tensor.shape[0], 1), self.coercive[1], dtype=torch.float32, device=points_tensor.device)
        return hx_tensor, hy_tensor

    def evaluate_current_density(self, points_tensor):
        return torch.full((points_tensor.shape[0], 1), self.current_density, dtype=torch.float32, device=points_tensor.device)