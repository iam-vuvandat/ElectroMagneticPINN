def evaluate_reluctivity(self, points_tensor, B_magnitude_tensor=None):
    if self.physics_config.bh_curve_data is None:
        self._compute_constant_bh_curve()
    try:
        return self.physics_config.bh_curve_data(points_tensor, B_magnitude_tensor)
    except TypeError:
        return self.physics_config.bh_curve_data(points_tensor)
