import os
import sys

# Đảm bảo Python nhận diện được các module trong thư mục gốc
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from geometry_engine.geometry import Geometry
from geometry_engine.segment.segment import Segment

def run_geometry_test():
    print("[*] Khởi tạo hệ thống hình học...")
    geo = Geometry()
    
    # Segment 1: Miền không khí nền bao trùm
    air_outline = [
        [-0.05, -0.05],
        [0.05, -0.05],
        [0.05, 0.05],
        [-0.05, 0.05]
    ]
    air_segment = Segment(
        outline=air_outline,
        material="air",
        relative_permeability=1.0,
        coercive=[0.0, 0.0],
        current=0.0
    )
    
    # Segment 2: Một khối nam châm chữ nhật nằm ở giữa
    mag_outline = [
        [-0.02, -0.02],
        [0.02, -0.02],
        [0.02, 0.02],
        [-0.02, 0.02]
    ]
    mag_segment = Segment(
        outline=mag_outline,
        material="magnet",
        relative_permeability=1.05,
        coercive=[800000.0, 0.0],
        current=0.0
    )
    
    # Thêm các segment vào mô hình
    geo.add_segment(air_segment)
    geo.add_segment(mag_segment)
    
    print(f"[*] Hệ thống ghi nhận tổng cộng {geo.get_segment_count()} segments.")
    print("[*] Đang kết xuất đồ thị điều kiện ban đầu (display method)...")
    
    # Gọi hàm display để kiểm tra 4 thông số vật lý
    geo.display(
        x_boundaries_tuple=(-0.05, 0.05), 
        y_boundaries_tuple=(-0.05, 0.05), 
        resolution=150
    )

if __name__ == "__main__":
    run_geometry_test()
