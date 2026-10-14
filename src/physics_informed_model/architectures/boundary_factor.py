def boundary_factor(self, xy):
    sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
    return sim_domain.boundary.compute_boundary_factor(xy, sim_domain.L0)
