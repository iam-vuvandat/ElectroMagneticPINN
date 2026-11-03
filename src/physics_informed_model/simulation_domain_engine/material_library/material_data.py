import torch
import numpy as np
from types import SimpleNamespace

VACUUM_RELUCTIVITY = 795774.715459  # nu0 = 1 / mu0
VACUUM_PERMEABILITY = 1.2566370614359173e-06  # mu0

class MaterialData:

    def __init__(self):
        self.data = [
            SimpleNamespace(
                name="air",
                relative_permeability=1.0,
                coercive=[0.0, 0.0],
                bh_curve_data=None
            ),
            SimpleNamespace(
                name="N30UH",
                relative_permeability=1.05,
                coercive=[0.0, 852000.0],
                bh_curve_data=None
            ),
            SimpleNamespace(
                name="NdFe30",
                relative_permeability=1.0445730167132,
                coercive=[0.0, 838000.0],
                bh_curve_data=None
            ),
            SimpleNamespace(
                name="steel_1008",
                relative_permeability=1000.0,  
                coercive=[0.0, 0.0],
                bh_curve_data=[
                    # B_points (Tesla) - Thép từ mềm phi tuyến cao (mu_r_max = 6953.37 tại 0.9T)
                    [0.0, 0.1, 0.5, 0.8, 0.9, 1.0, 1.3, 1.5, 1.9, 2.4],
                    # H_points (A/m)
                    [0.0, 34.8, 67.4, 91.8, 103.0, 119.0, 250.0, 1180.0, 25000.0, 229580.0]
                ]
            )
        ]

    def get_material(self, name: str) -> SimpleNamespace:
        
        for mat in self.data:
            if mat.name.lower() == name.lower():
                return mat
        raise KeyError(f"Lỗi: Không tìm thấy vật liệu '{name}' trong MaterialData!")

    def __getitem__(self, index):
        """Cho phép truy cập dạng index: mat_data[0]"""
        return self.data[index]

    def __len__(self):
        """Trả về số lượng vật liệu trong kho"""
        return len(self.data)