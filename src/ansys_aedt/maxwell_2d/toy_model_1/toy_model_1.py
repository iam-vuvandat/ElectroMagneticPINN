from types import SimpleNamespace
from .configuration import Configuration
from .create_model import create_model
from .run_simulation import run_simulation
from .export_flux_density import export_flux_density
from .plot_flux_density import plot_flux_density

class ToyModel1:
    def __init__(self):
        self.configuration = Configuration()
        self.m2d = None
        self.record = SimpleNamespace(
            flux_density_field=None
        )

    def create_model(self):
        return create_model(toy_model_obj=self)

    def run_simulation(self):
        return run_simulation(toy_model_obj=self)

    def export_flux_density(self, export_path=None, min_bounds=None, max_bounds=None, grid_step=None):
        return export_flux_density(
            toy_model_obj=self, 
            export_path=export_path,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            grid_step=grid_step
        )

    def plot_flux_density(
        self, 
        fld_path=None, 
        output_png_path=None,
        save_path=None,
        save_image_path=None,
        smooth="cubic", 
        target_resolution=1500, 
        sigma_smooth=0.0,
        show=True,
        figsize=(6.4, 4.8),
        dpi=100, 
        vmin=0.0,
        vmax=2.0,
        cmap="ansys_fem",
        num_divisions=100,
        **kwargs
    ):
        return plot_flux_density(
            toy_model_obj=self,
            fld_path=fld_path, 
            output_png_path=output_png_path,
            save_path=save_path,
            save_image_path=save_image_path,
            smooth=smooth, 
            target_resolution=target_resolution, 
            sigma_smooth=sigma_smooth,
            show=show,
            figsize=figsize,
            dpi=dpi,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            num_divisions=num_divisions,
            **kwargs
        )