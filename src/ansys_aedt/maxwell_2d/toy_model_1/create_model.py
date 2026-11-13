import os
from pathlib import Path
from ansys.aedt.core import Maxwell2d
from .reset_ansys_environment import reset_ansys_environment

def _outline_to_rect_params(outline):
    xs = [pt[0] for pt in outline]
    ys = [pt[1] for pt in outline]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Quy đổi từ mét sang mm cho Ansys FEM
    x_start_str = f"{x_min * 1000.0}mm"
    y_start_str = f"{y_min * 1000.0}mm"
    width_str = f"{(x_max - x_min) * 1000.0}mm"
    height_str = f"{(y_max - y_min) * 1000.0}mm"
    return x_start_str, y_start_str, width_str, height_str

def create_model(toy_model_obj):
    print("\033[94mIn function create_model.\033[0m")
    print("\033[94m{\033[0m")

    reset_ansys_environment()
    
    config = toy_model_obj.configuration
    version = str(config.version)

    m2d = Maxwell2d(
        design=config.name,
        solution_type=config.solution_type,
        version=version,
        new_desktop=True,
        non_graphical=False
    )
    print(f"\033[94mIn function create_model: Initialized Maxwell 2D design '{config.name}'.\033[0m")

    mat_data = toy_model_obj.simulation_domain.material_data if hasattr(toy_model_obj, "simulation_domain") else None
    if mat_data is not None:
        ndfe30_mat = mat_data.get_material("NdFe30")
        hc_val = ndfe30_mat.coercive[1]
    else:
        hc_val = 838000.0

    oDefinitionManager = m2d.odefinition_manager

    oDefinitionManager.AddMaterial(
        [
            "NAME:NdFe30 - Copy",
            "CoordinateSystemType:=", "Cartesian",
            "BulkOrSurfaceType:=", 1,
            ["NAME:PhysicsTypes", "set:=", ["Electromagnetic", "Thermal", "Structural"]],
            [
                "NAME:AttachedData",
                [
                    "NAME:MatAppearanceData",
                    "property_data:=", "appearance_data",
                    "Red:=", 204,
                    "Green:=", 204,
                    "Blue:=", 204
                ]
            ],
            "permittivity:=", "1",
            "permeability:=", "1.0445730167132",
            "conductivity:=", "625000",
            "dielectric_loss_tangent:=", "0",
            "magnetic_loss_tangent:=", "0",
            [
                "NAME:magnetic_coercivity",
                "property_type:=", "VectorProperty",
                "Magnitude:=", f"-{hc_val}A_per_meter",
                "DirComp1:=", "0",
                "DirComp2:=", "1",
                "DirComp3:=", "0"
            ],
            "thermal_conductivity:=", "0",
            "saturation_mag:=", "0gauss",
            "lande_g_factor:=", "2",
            "delta_H:=", "0Oe",
            "mass_density:=", "7550",
            "youngs_modulus:=", "147000000000",
            [
                "NAME:thermal_expansion_coefficient",
                "property_type:=", "AnisoProperty",
                "unit:=", "",
                "component1:=", "3e-06",
                "component2:=", "-5e-06",
                "component3:=", "-5e-06"
            ]
        ]
    )

    oDefinitionManager.AddMaterial(
        [
            "NAME:NdFe30 - Copy (2)",
            "CoordinateSystemType:=", "Cartesian",
            "BulkOrSurfaceType:=", 1,
            ["NAME:PhysicsTypes", "set:=", ["Electromagnetic", "Thermal", "Structural"]],
            [
                "NAME:AttachedData",
                [
                    "NAME:MatAppearanceData",
                    "property_data:=", "appearance_data",
                    "Red:=", 204,
                    "Green:=", 204,
                    "Blue:=", 204
                ]
            ],
            "permittivity:=", "1",
            "permeability:=", "1.0445730167132",
            "conductivity:=", "625000",
            "dielectric_loss_tangent:=", "0",
            "magnetic_loss_tangent:=", "0",
            [
                "NAME:magnetic_coercivity",
                "property_type:=", "VectorProperty",
                "Magnitude:=", f"-{hc_val}A_per_meter",
                "DirComp1:=", "0",
                "DirComp2:=", "-1",
                "DirComp3:=", "0"
            ],
            "thermal_conductivity:=", "0",
            "saturation_mag:=", "0gauss",
            "lande_g_factor:=", "2",
            "delta_H:=", "0Oe",
            "mass_density:=", "7550",
            "youngs_modulus:=", "147000000000",
            [
                "NAME:thermal_expansion_coefficient",
                "property_type:=", "AnisoProperty",
                "unit:=", "",
                "component1:=", "3e-06",
                "component2:=", "-5e-06",
                "component3:=", "-5e-06"
            ]
        ]
    )

    # 3. Tính toán hình học từ outline và vẽ trong Ansys FEM
    yoke_x, yoke_y, yoke_w, yoke_h = _outline_to_rect_params(config.geometry.yoke.outline)
    iron_yoke = m2d.modeler.create_rectangle(
        origin=[yoke_x, yoke_y, "0mm"],
        sizes=[yoke_w, yoke_h],
        name="iron_yoke",
        material=config.materials.yoke_material
    )

    ml_x, ml_y, ml_w, ml_h = _outline_to_rect_params(config.geometry.magnet_left.outline)
    mag_l = m2d.modeler.create_rectangle(
        origin=[ml_x, ml_y, "0mm"],
        sizes=[ml_w, ml_h],
        name="magnet_left",
        material=config.materials.magnet_material_left
    )

    mr_x, mr_y, mr_w, mr_h = _outline_to_rect_params(config.geometry.magnet_right.outline)
    mag_r = m2d.modeler.create_rectangle(
        origin=[mr_x, mr_y, "0mm"],
        sizes=[mr_w, mr_h],
        name="magnet_right",
        material=config.materials.magnet_material_right
    )

    reg_x, reg_y, reg_w, reg_h = _outline_to_rect_params(config.geometry.region.outline)
    region = m2d.modeler.create_rectangle(
        origin=[reg_x, reg_y, "0mm"],
        sizes=[reg_w, reg_h],
        name="region",
        material=config.materials.region_material
    )

    m2d.assign_balloon(region.edges, boundary="Balloon1")

    m2d.mesh.assign_length_mesh(
        assignment=[iron_yoke.name, mag_l.name, mag_r.name, region.name],
        maximum_length=config.analysis.max_element_length,
        name="Mesh_1mm"
    )

    setup = m2d.create_setup(name=config.analysis.setup_name)
    setup.props["MaximumPasses"] = config.analysis.max_passes
    setup.props["MinimumPasses"] = config.analysis.min_passes
    setup.props["PercentError"] = config.analysis.percent_error

    print(f"\033[94mIn function create_model: Successfully created 2D geometry from outlines, Mesh ({config.analysis.max_element_length}), Balloon boundary, and Setup '{config.analysis.setup_name}'.\033[0m")
    print("\033[94m}\033[0m\n")

    toy_model_obj.m2d = m2d
    return toy_model_obj