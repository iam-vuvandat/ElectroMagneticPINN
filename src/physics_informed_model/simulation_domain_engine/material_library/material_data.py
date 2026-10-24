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
                    # B_points (Tesla)
                    [0.0, 0.2402, 0.8654, 1.1106, 1.2458, 1.331, 1.5, 1.6, 1.683, 
                     1.741, 1.78, 1.905, 2.025, 2.085, 2.13, 2.165, 2.28, 2.485, 2.5851],
                    # H_points (A/m)
                    [0.0, 159.2, 318.3, 477.5, 636.6, 795.8, 1591.5, 3183.1, 4774.6, 
                     6366.2, 7957.7, 15915.5, 31831.0, 47746.5, 63662.0, 79577.5, 
                     159155.0, 318310.0, 397887.0]
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