import os
import imageio

def generate_gif(visualizer_instance):
    cfg = visualizer_instance.electro_magnetic_pinn.visualization_config
    output_filename = getattr(cfg, 'gif_filename', 'training_process.gif')
    fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
    if not visualizer_instance.frame_paths:
        return
        
    full_gif_path = os.path.join(visualizer_instance.output_directory, output_filename)
    images = [imageio.imread(f) for f in visualizer_instance.frame_paths if os.path.exists(f)]
    imageio.mimsave(full_gif_path, images, fps=fps, loop=0)
