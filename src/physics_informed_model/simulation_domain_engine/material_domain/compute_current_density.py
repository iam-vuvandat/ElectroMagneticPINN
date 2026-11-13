def compute_current_density(self):
    if callable(self.physics_config.current):
        self.current_density = self.physics_config.current
        return self.current_density
    area = self.section_area 
    if area > 0.0:
        self.current_density = float(self.physics_config.current) / area
    else:
        self.current_density = 0.0
    return self.current_density
