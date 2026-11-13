from types import SimpleNamespace

from shapely import boundary

class Configuation:
    def __init__(self):
        self.name = "reluctance_synchronous_motor_0001"
        self.version = 2025.2
        self.machine = SimpleNamespace(
            source_type = "AC",
            structure = "Inner Rotor",
            use_minimum_setting = False,
            stator_core_type = "SLOT_AC",
            rotor_core_type = "NONS_RELU",
            stator = SimpleNamespace(
                number_of_poles = 10,
                number_of_slots = 15,
                circuit_type = "Y3",
                slot_type = 1,
                position_control = False,
                core = SimpleNamespace(
                    outer_diameter = 150,
                    inner_diameter = 70,
                    length = 70,
                    stacking_factor = 0.95,
                    steel_type = "steel_1008",
                    magnetic_press_board = False,
                    skew_width = 0,
                    slot = SimpleNamespace(
                        auto_design = False,
                        parallel_tooth = False,
                        hs0 = 1,
                        hs2 = 1,
                        bs0 = 1,
                        bs1 = 1,
                        bs2 = 1
                    )
                ),
                winding = SimpleNamespace(
                    winding_layers = 2,
                    winding_type = "Whole-Coiled",
                    parallel_branches = 1,
                    conductors_per_slot = 10,
                    coil_pitch = 1,
                    number_of_strands = 1,
                    wire_wrap = 1,
                    conductor_type = "copper"
                )
            ),
            rotor = SimpleNamespace(
                number_of_poles = 10,
                core = SimpleNamespace(
                    outer_diameter = 69,
                    inner_diameter = 20,
                    length = 70,
                    stacking_factor = 1.0,
                    steel_type = "steel_1008",
                    pole_type = 4,
                    pole = SimpleNamespace(
                        barriers_per_pole = 2,
                        h = 2,
                        w = 2,
                        r = 1,
                        r0 = 1,
                        rb = 15,
                        b0 = 2,
                        y0 = 2,
                        barrier_auto_arrangement = True
                    )
                )
            ),
            shaft = SimpleNamespace(
                magnetic_shaft = False,
                frictional_loss = 1,
                windage_loss_or_power = 1,
                reference_speed = 3000
            )
        )
        self.analysis = SimpleNamespace(
            setup_name = "Setup1",
            enabled = True,
            import_mesh = False,
            rated_output_power = 3000,
            rated_voltage = 300,
            rated_speed = 3000,
            operating_temperature = 60,
            operation_type = "Motor",
            load_type = "ConstSpeed",
            rated_power_factor = 0.8,
            frequency = 60,
            capacitive_power_factor = False
        )
        self.maxwell_2d = SimpleNamespace(
            component_3d = None,
            model = None,
            boundary = None,
            excitation = SimpleNamespace(
                phase_a = SimpleNamespace(
                    type = "Current",
                    current = "0A"
                )
            )
        )