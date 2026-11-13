from simulation_domain_engine.material_domain.material_domain import MaterialDomain

def build_background_domain(self, domain_name="background", training_config=None, network_config=None):
    if self.background_domain in self.domains_list:
        self.domains_list.remove(self.background_domain)

    x_min, x_max = self.boundary.geometry_config.x_boundaries_tuple
    y_min, y_max = self.boundary.geometry_config.y_boundaries_tuple
    
    box_outline = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]

    bg_domain = MaterialDomain(domain_name=domain_name, simulation_domain=self)
    bg_domain.geometry_config.outline = box_outline
    bg_domain.physics_config.relative_permeability = 1.0
    bg_domain.physics_config.coercive = [0.0, 0.0]
    bg_domain.physics_config.current = 0.0
    bg_domain.training_config.number_of_interior_points = 4000
    bg_domain.training_config.number_of_surface_points = 0

    if training_config is not None:
        for key, value in vars(training_config).items():
            setattr(bg_domain.training_config, key, value)

    if network_config is not None:
        for key, value in vars(network_config).items():
            setattr(bg_domain.network_config, key, value)

    bg_domain.build_domain()

    total_box_area = (x_max - x_min) * (y_max - y_min)
    total_sub_material_area = sum(d.section_area for d in self.domains_list)
    bg_domain.section_area = max(0.0, total_box_area - total_sub_material_area)

    self.background_domain = bg_domain
    self.domains_list.append(self.background_domain)

    for idx, dom in enumerate(self.domains_list):
        dom.domain_id = idx + 1

    return self.background_domain
