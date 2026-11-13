def set_boundary(self, boundary_object):
    boundary_object.simulation_domain = self
    self.boundary = boundary_object
    return self
