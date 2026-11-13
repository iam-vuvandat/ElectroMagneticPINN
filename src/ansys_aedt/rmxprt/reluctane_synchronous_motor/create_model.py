import subprocess
import time
from pathlib import Path
from ansys.aedt.core import Rmxprt, Maxwell2d
from types import SimpleNamespace

from .reset_ansys_environment import reset_ansys_environment

PROP_MAPPING = {
    "source_type": ("Source Type", "Value:=", ""),
    "structure": ("Structure", "Value:=", ""),
    "use_minimum_setting": ("Use Minimum Setting", "Value:=", ""),
    "stator_core_type": ("Stator/Core Type", "Value:=", ""),
    "rotor_core_type": ("Rotor/Core Type", "Value:=", ""),
    
    "number_of_poles": ("Number of Poles", "Value:=", ""),
    "number_of_slots": ("Number of Slots", "Value:=", ""),
    "circuit_type": ("Circuit Type", "CircuitType:=", ""),
    "slot_type": ("Slot Type", "SlotType:=", ""),
    "position_control": ("Position Control", "Value:=", ""),
    
    "outer_diameter": ("Outer Diameter", "Value:=", "mm"),
    "inner_diameter": ("Inner Diameter", "Value:=", "mm"),
    "length": ("Length", "Value:=", "mm"),
    "stacking_factor": ("Stacking Factor", "Value:=", ""),
    "steel_type": ("Steel Type", "Material:=", ""),
    "magnetic_press_board": ("Magnetic Press Board", "Value:=", ""),
    "skew_width": ("Skew Width", "Value:=", "deg"),
    
    "auto_design": ("Auto Design", "Value:=", ""),
    "parallel_tooth": ("Parallel Tooth", "Value:=", ""),
    "hs0": ("Hs0", "Value:=", "mm"),
    "hs2": ("Hs2", "Value:=", "mm"),
    "bs0": ("Bs0", "Value:=", "mm"),
    "bs1": ("Bs1", "Value:=", "mm"),
    "bs2": ("Bs2", "Value:=", "mm"),
    
    "winding_layers": ("Winding Layers", "Value:=", ""),
    "winding_type": ("Winding Type", "WindingType:=", ""),
    "parallel_branches": ("Parallel Branches", "Value:=", ""),
    "conductors_per_slot": ("Conductors per Slot", "Value:=", ""),
    "coil_pitch": ("Coil Pitch", "Value:=", ""),
    "number_of_strands": ("Number of Strands", "Value:=", ""),
    "wire_wrap": ("Wire Wrap", "Value:=", "mm"),
    "conductor_type": ("Conductor Type", "Material:=", ""),
    
    "pole_type": ("Pole Type", "PoleType:=", ""),
    
    "barriers_per_pole": ("Barriers per Pole", "Value:=", ""),
    "h": ("H", "Value:=", "mm"),
    "w": ("W", "Value:=", "mm"),
    "r": ("R", "Value:=", "mm"),
    "r0": ("R0", "Value:=", "mm"),
    "rb": ("Rb", "Value:=", "mm"),
    "b0": ("B0", "Value:=", "mm"),
    "y0": ("Y0", "Value:=", "mm"),
    "barrier_auto_arrangement": ("Barrier Auto Arrangement", "Value:=", ""),
    
    "magnetic_shaft": ("Magnetic Shaft", "Value:=", ""),
    "frictional_loss": ("Frictional Loss", "Value:=", "W"),
    "windage_loss_or_power": ("Windage Loss or Power", "Value:=", "W"),
    "reference_speed": ("Reference Speed", "Value:=", "rpm")
}

def apply_properties(oEditor, tab_name, prop_server, config_namespace):
    print("\033[94mIn function apply_properties.\033[0m")
    print("\033[94m{\033[0m")
    
    for key, val in vars(config_namespace).items():
        if isinstance(val, SimpleNamespace) or val is None:
            continue
            
        if key in PROP_MAPPING:
            ansys_name, ansys_val_type, unit = PROP_MAPPING[key]
            
            if isinstance(val, bool):
                formatted_val = val
            elif unit:
                formatted_val = f"{val}{unit}"
            else:
                formatted_val = str(val) if isinstance(val, (int, float)) else val

            oEditor.ChangeProperty(
                [
                    "NAME:AllTabs",
                    [
                        "NAME:" + tab_name,
                        ["NAME:PropServers", prop_server],
                        ["NAME:ChangedProps", ["NAME:" + ansys_name, ansys_val_type, formatted_val]]
                    ]
                ]
            )
            time.sleep(0.1)
            
    print(f"\033[94mIn function apply_properties: Applied properties for {prop_server}.\033[0m")
    print("\033[94m}\033[0m")
    print("\033[94m\033[0m")

def create_model(reluctance_synchronous_motor):
    print("\033[94mIn function create_model.\033[0m")
    print("\033[94m{\033[0m")
    
    reset_ansys_environment()
    print("\033[94mIn function create_model: Reset Ansys environment.\033[0m")
    
    configuation = reluctance_synchronous_motor.configuation
    name = configuation.name
    version = str(configuation.version)

    rmxprt = Rmxprt(version=version, new_desktop=True, non_graphical=False)
    print("\033[94mIn function create_model: Initialized Rmxprt object.\033[0m")
    
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir
    while not (project_root / "src").exists() and project_root.parent != project_root:
        project_root = project_root.parent
        
    save_path = project_root / "data" / "temp"
    save_path.mkdir(parents=True, exist_ok=True)

    if not name.endswith(".aedt"):
        project_name = name + ".aedt"
    else:
        project_name = name

    project_file = save_path / project_name
    rmxprt.save_project(str(project_file))
    print(f"\033[94mIn function create_model: Saved project to {project_file}.\033[0m")

    rmxprt.oproject.InsertDesign(
        "RMxprt", 
        name, 
        "Model Creation Reluctance Synchronous Machine", 
        ""
    )
    rmxprt.set_active_design(name)

    oDesign = rmxprt.odesign
    oEditor = oDesign.SetActiveEditor("Machine")
    
    apply_properties(oEditor, "Machine", "Machine", configuation.machine)
    
    apply_properties(oEditor, "Machine", "Stator", configuation.machine.stator)
    apply_properties(oEditor, "Machine", "Stator:Core", configuation.machine.stator.core)
    apply_properties(oEditor, "Slot", "Stator:Core:Slot", configuation.machine.stator.core.slot)
    apply_properties(oEditor, "Winding", "Stator:Winding", configuation.machine.stator.winding)
    
    apply_properties(oEditor, "Machine", "Rotor", configuation.machine.rotor)
    apply_properties(oEditor, "Machine", "Rotor:Core", configuation.machine.rotor.core)
    apply_properties(oEditor, "Pole", "Rotor:Core:Pole", configuation.machine.rotor.core.pole)
    
    apply_properties(oEditor, "Shaft", "Shaft", configuation.machine.shaft)
    print("\033[94mIn function create_model: Applied all machine properties.\033[0m")

    analysis_config = configuation.analysis
    setup_name = analysis_config.setup_name
    
    oModule = oDesign.GetModule("AnalysisSetup")
    oModule.InsertSetup("GRM", 
        [
            f"NAME:{setup_name}",
            "Enabled:=", analysis_config.enabled,
            [
                "NAME:MeshLink",
                "ImportMesh:=", analysis_config.import_mesh
            ],
            "RatedOutputPower:=", f"{analysis_config.rated_output_power}W",
            "RatedVoltage:=", f"{analysis_config.rated_voltage}V",
            "RatedSpeed:=", f"{analysis_config.rated_speed}rpm",
            "OperatingTemperature:=", f"{analysis_config.operating_temperature}cel",
            "OperationType:=", analysis_config.operation_type,
            "LoadType:=", analysis_config.load_type,
            "RatedPowerFactor:=", str(analysis_config.rated_power_factor),
            "Frequency:=", f"{analysis_config.frequency}Hz",
            "CapacitivePowerFactor:=", analysis_config.capacitive_power_factor
        ]
    )
    print("\033[94mIn function create_model: Configured Analysis Setup.\033[0m")
    
    designs_before = list(rmxprt.oproject.GetTopDesignList())
    
    oModule.CreateMaxwellModel(setup_name)
    print("\033[94mIn function create_model: Created Maxwell 2D model.\033[0m")

    time.sleep(1)

    designs_after = list(rmxprt.oproject.GetTopDesignList())
    new_designs = [d for d in designs_after if d not in designs_before]
    m2d_design_name = new_designs[0] if new_designs else "Maxwell2DDesign1"
    
    designs_to_keep = [name, m2d_design_name]
    for d in designs_after:
        if d not in designs_to_keep:
            rmxprt.oproject.DeleteDesign(d)
            print(f"\033[94mIn function create_model: Deleted redundant design '{d}'.\033[0m")

    m2d = Maxwell2d(
        version=version,
        project=rmxprt.project_name,
        design=m2d_design_name,
        non_graphical=False
    )
    print(f"\033[94mIn function create_model: Connected to Maxwell 2D object '{m2d_design_name}'.\033[0m")

    reluctance_synchronous_motor.rmxprt = rmxprt
    reluctance_synchronous_motor.m2d = m2d
    
    print("\033[94m}\033[0m")
    print("\033[94m\033[0m")
    
    return reluctance_synchronous_motor