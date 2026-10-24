def assign_material(self, material_name="air"):
    if self.simulation_domain is None or not hasattr(self.simulation_domain, 'material_data'):
        raise RuntimeError(f"Lỗi: Domain '{self.domain_name}' chưa được gán vào SimulationDomain nên không thể tra cứu vật liệu.")

    mat = self.simulation_domain.material_data.get_material(material_name)
    self.physics_config.relative_permeability = mat.relative_permeability
    self.physics_config.coercive = mat.coercive
    self.physics_config.bh_curve_data = mat.bh_curve_data
    
    print(f"[*] Domain '{self.domain_name}': Đã gán thành công vật liệu '{mat.name}'.")
    return self
