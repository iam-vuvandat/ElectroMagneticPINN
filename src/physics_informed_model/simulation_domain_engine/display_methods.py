from simulation_domain_engine.domain_visualizer import display_simulation_domain

def plot_problem_definition(self, resolution=100):
    display_simulation_domain(self, self.x_boundaries_tuple, self.y_boundaries_tuple, resolution)

def display(self, x_boundaries_tuple=None, y_boundaries_tuple=None, resolution=150):
    xb = x_boundaries_tuple if x_boundaries_tuple is not None else self.x_boundaries_tuple
    yb = y_boundaries_tuple if y_boundaries_tuple is not None else self.y_boundaries_tuple
    display_simulation_domain(self, xb, yb, resolution)
