from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

def compute_constant_bh_curve(self):
    mu_r = float(getattr(self.physics_config, 'relative_permeability', 1.0))
    nu_val = VACUUM_RELUCTIVITY / mu_r


    b_samples = [0.0, 10.0]
    h_samples = [0.0, 10.0 * nu_val]

    self.physics_config.bh_curve_data = [b_samples, h_samples]
    return self.physics_config.bh_curve_data