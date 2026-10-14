import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from simulation_domain_engine.simulation_domain import SimulationDomain
from simulation_domain_engine.material_domain.material_domain import MaterialDomain

def run_geometry_test():
    geo = SimulationDomain()
    
    air_domain = MaterialDomain(domain_name="air")
    air_domain.geometry_config.outline = [
        [-0.05, -0.05],
        [0.05, -0.05],
        [0.05, 0.05],
        [-0.05, 0.05]
    ]
    air_domain.physics_config.relative_permeability = 1.0
    air_domain.build_domain()
    
    mag_domain = MaterialDomain(domain_name="magnet")
    mag_domain.geometry_config.outline = [
        [-0.02, -0.02],
        [0.02, -0.02],
        [0.02, 0.02],
        [-0.02, 0.02]
    ]
    mag_domain.physics_config.relative_permeability = 1.05
    mag_domain.physics_config.coercive = [800000.0, 0.0]
    mag_domain.build_domain()
    
    geo.add_material_domain(air_domain)
    geo.add_material_domain(mag_domain)
    
    geo.display(
        x_boundaries_tuple=(-0.05, 0.05), 
        y_boundaries_tuple=(-0.05, 0.05), 
        resolution=150
    )
    
if __name__ == "__main__":
    run_geometry_test()
