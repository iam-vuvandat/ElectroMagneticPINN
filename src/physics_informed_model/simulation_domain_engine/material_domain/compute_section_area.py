def compute_section_area(self):
    if self.geometry_config.outline is None or len(self.geometry_config.outline) < 3:
        self.section_area = 0.0
        return self.section_area
    x = [p[0] for p in self.geometry_config.outline]
    y = [p[1] for p in self.geometry_config.outline]
    self.section_area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    return self.section_area
