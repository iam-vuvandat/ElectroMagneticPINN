import os
from pathlib import Path
from types import SimpleNamespace

class Configuration:
    def __init__(self):
        self.name = "toy_model_1"
        self.version = "2025.2"
        self.solution_type = "Magnetostatic"
        
        self.geometry = SimpleNamespace(
            yoke=SimpleNamespace(
                outline=[[-0.06, -0.03], [0.06, -0.03], [0.06, 0.0], [-0.06, 0.0]]
            ),
            magnet_left=SimpleNamespace(
                outline=[[-0.05, 0.0], [-0.005, 0.0], [-0.005, 0.018], [-0.05, 0.018]],
                coercive_direction=(0, 1)
            ),
            magnet_right=SimpleNamespace(
                outline=[[0.005, 0.0], [0.05, 0.0], [0.05, 0.018], [0.005, 0.018]],
                coercive_direction=(0, -1)
            ),
            region=SimpleNamespace(
                outline=[[-0.12, -0.12], [0.12, -0.12], [0.12, 0.12], [-0.12, 0.12]]
            )
        )
        
        self.materials = SimpleNamespace(
            yoke_material="steel_1008",
            magnet_material_left="NdFe30 - Copy",
            magnet_material_right="NdFe30 - Copy (2)",
            region_material="vacuum"
        )

        default_b_fld_path = str(Path(__file__).resolve().parent / "flux_density.fld")

        self.analysis = SimpleNamespace(
            setup_name="Setup1",
            max_passes=10,
            min_passes=2,
            percent_error=1.0,
            max_element_length="0.025mm",
            b_fld_path=default_b_fld_path,
            grid_min=["-120mm", "-120mm", "0mm"],
            grid_max=["120mm", "120mm", "0mm"],
            grid_step=["1mm", "1mm", "0mm"]
        )