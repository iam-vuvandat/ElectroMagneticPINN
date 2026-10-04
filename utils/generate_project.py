import os

def generate_u_shape_template():
    base_dir = os.getcwd()
    
    template_dir = os.path.join(
        base_dir, 
        'src', 
        'physics_informed_model', 
        'geometry_engine', 
        'templates'
    )
    
    os.makedirs(template_dir, exist_ok=True)
    
    init_file = os.path.join(template_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('')
            
    u_shape_code = """import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
from src.physics_informed_model.geometry_engine.geometry import Polygon

class UShapeMagnetTemplate:
    def __init__(self, width, height, thickness, air_box_size=5.0, material="magnet"):
        self.width = width
        self.height = height
        self.thickness = thickness
        self.air_box_size = air_box_size
        self.material = material

    def build(self):
        air_box = Polygon().set_material("air").set_vertices([
            [-self.air_box_size/2, self.air_box_size/2],
            [self.air_box_size/2, self.air_box_size/2],
            [self.air_box_size/2, -self.air_box_size/2],
            [-self.air_box_size/2, -self.air_box_size/2]
        ])
        
        outer_rect = Polygon().set_vertices([
            [-self.width/2, self.height/2],
            [self.width/2, self.height/2],
            [self.width/2, -self.height/2],
            [-self.width/2, -self.height/2]
        ])
        
        inner_rect = Polygon().set_vertices([
            [-(self.width/2 - self.thickness), self.height/2],
            [(self.width/2 - self.thickness), self.height/2],
            [(self.width/2 - self.thickness), -self.height/2 + self.thickness],
            [-(self.width/2 - self.thickness), -self.height/2 + self.thickness]
        ])
        
        magnet_u_shape = outer_rect - inner_rect
        magnet_u_shape.set_material(self.material)
        
        air_domain = air_box - magnet_u_shape
        
        return air_domain, magnet_u_shape, air_box

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    template = UShapeMagnetTemplate(width=2.0, height=3.0, thickness=0.5, air_box_size=5.0)
    air_domain, magnet_u_shape, air_box = template.build()
    
    x = torch.linspace(-3, 3, 200)
    y = torch.linspace(-3, 3, 200)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    sdf_magnet = magnet_u_shape.compute_sdf(xy)
    mask_magnet = sdf_magnet <= 0
    
    xy_np = xy.detach().numpy()
    mask_np = mask_magnet.detach().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(xy_np[mask_np, 0], xy_np[mask_np, 1], c='red', label='Magnet', s=2)
    plt.scatter(xy_np[~mask_np, 0], xy_np[~mask_np, 1], c='lightblue', alpha=0.3, label='Air', s=2)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.legend()
    plt.title("U-Shape Magnet Geometry")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
"""

    template_file_path = os.path.join(template_dir, 'u_shape_magnet_template.py')
    with open(template_file_path, 'w', encoding='utf-8') as f:
        f.write(u_shape_code)
        
if __name__ == "__main__":
    generate_u_shape_template()