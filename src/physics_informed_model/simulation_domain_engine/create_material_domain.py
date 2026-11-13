from simulation_domain_engine.material_domain.material_domain import MaterialDomain

def create_material_domain(
    self,
    domain_name: str,
    material_name: str = None,
    outline: list = None,
    coercive_direction: tuple = None
) -> MaterialDomain:
    domain = MaterialDomain(domain_name=domain_name, simulation_domain=self)
    if outline is not None:
        domain.geometry_config.outline = outline
    if material_name is not None:
        domain.assign_material(material_name)
    if coercive_direction is not None:
        domain.assign_coercive_direction(coercive_direction)
    self.add_material_domain(domain)
    return domain
