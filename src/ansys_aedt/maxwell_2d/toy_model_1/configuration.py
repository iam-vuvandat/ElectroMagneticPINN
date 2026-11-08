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
                x_start="-60mm",
                y_start="-30mm",
                width="120mm",
                height="30mm"
            ),
            magnet_left=SimpleNamespace(
                x_start="-50mm",
                y_start="0mm",
                width="45mm",
                height="18mm",
                coercive_direction=(0, 1)
            ),
            magnet_right=SimpleNamespace(
                x_start="5mm",
                y_start="0mm",
                width="45mm",
                height="18mm",
                coercive_direction=(0, -1)
            ),
            region=SimpleNamespace(
                x_start="-120mm",
                y_start="-120mm",
                width="240mm",
                height="240mm"
            )
        )
        
        self.materials = SimpleNamespace(
            yoke_material="steel_1008",
            magnet_material_left="NdFe30 - Copy",
            magnet_material_right="NdFe30 - Copy (2)",
            region_material="vacuum",
            hc_magnitude=838000.0
        )

        default_b_fld_path = str(Path(__file__).resolve().parent / "flux_density.fld")

        self.analysis = SimpleNamespace(
            setup_name="Setup1",
            max_passes=10,
            min_passes=2,
            percent_error=1.0,
            max_element_length="0.25mm",
            b_fld_path=default_b_fld_path,
            grid_min=["-120mm", "-120mm", "0mm"],
            grid_max=["120mm", "120mm", "0mm"],
            grid_step=["0.1mm", "0.1mm", "0mm"]
        )
