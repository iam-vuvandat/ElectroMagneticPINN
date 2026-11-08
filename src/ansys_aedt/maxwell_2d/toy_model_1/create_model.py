import os
from pathlib import Path
from ansys.aedt.core import Maxwell2d
from .reset_ansys_environment import reset_ansys_environment

def create_model(toy_model_obj):
    print("\033[94mIn function create_model.\033[0m")
    print("\033[94m{\033[0m")

    reset_ansys_environment()
    
    config = toy_model_obj.configuration
    version = str(config.version)

    # 1. Khởi tạo Desktop Session
    m2d = Maxwell2d(
        design=config.name,
        solution_type=config.solution_type,
        version=version,
        new_desktop=True,
        non_graphical=False
    )
    print(f"\033[94mIn function create_model: Initialized Maxwell 2D design '{config.name}'.\033[0m")

    # 2. Định nghĩa vật liệu bản sao NdFe30 bằng oDefinitionManager.AddMaterial
    oDefinitionManager = m2d.odefinition_manager

    # NdFe30 - Copy (Hc hướng +Y)
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
                "Magnitude:=", f"-{config.materials.hc_magnitude}A_per_meter",
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

    # NdFe30 - Copy (2) (Hc hướng -Y)
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
                "Magnitude:=", f"-{config.materials.hc_magnitude}A_per_meter",
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

    # 3. Dựng 4 hình chữ nhật và gán vật liệu
    iron_yoke = m2d.modeler.create_rectangle(
        origin=[config.geometry.yoke.x_start, config.geometry.yoke.y_start, "0mm"],
        sizes=[config.geometry.yoke.width, config.geometry.yoke.height],
        name="iron_yoke",
        material=config.materials.yoke_material
    )

    mag_l = m2d.modeler.create_rectangle(
        origin=[config.geometry.magnet_left.x_start, config.geometry.magnet_left.y_start, "0mm"],
        sizes=[config.geometry.magnet_left.width, config.geometry.magnet_left.height],
        name="magnet_left",
        material=config.materials.magnet_material_left
    )

    mag_r = m2d.modeler.create_rectangle(
        origin=[config.geometry.magnet_right.x_start, config.geometry.magnet_right.y_start, "0mm"],
        sizes=[config.geometry.magnet_right.width, config.geometry.magnet_right.height],
        name="magnet_right",
        material=config.materials.magnet_material_right
    )

    region = m2d.modeler.create_rectangle(
        origin=[config.geometry.region.x_start, config.geometry.region.y_start, "0mm"],
        sizes=[config.geometry.region.width, config.geometry.region.height],
        name="region",
        material=config.materials.region_material
    )

    # 4. Gán điều kiện biên Balloon
    m2d.assign_balloon(region.edges, boundary="Balloon1")

    # 5. Cài đặt chiều dài element tối đa trong FEM = 1mm
    m2d.mesh.assign_length_mesh(
        assignment=[iron_yoke.name, mag_l.name, mag_r.name, region.name],
        maximum_length=config.analysis.max_element_length,
        name="Mesh_1mm"
    )

    # 6. Khởi tạo Analyze Setup
    setup = m2d.create_setup(name=config.analysis.setup_name)
    setup.props["MaximumPasses"] = config.analysis.max_passes
    setup.props["MinimumPasses"] = config.analysis.min_passes
    setup.props["PercentError"] = config.analysis.percent_error

    print(f"\033[94mIn function create_model: Successfully created 2D geometry, Mesh (1mm), Balloon boundary, and Setup '{config.analysis.setup_name}'.\033[0m")
    print("\033[94m}\033[0m\n")

    toy_model_obj.m2d = m2d
    return toy_model_obj
