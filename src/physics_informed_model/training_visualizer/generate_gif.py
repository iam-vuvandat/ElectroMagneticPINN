# src/physics_informed_model/training_visualizer/generate_gif.py
from IPython.display import Image, display
import os
import imageio

def generate_gif(visualizer_instance):
    cfg = visualizer_instance.electro_magnetic_pinn.visualization_config
    output_filename = getattr(cfg, 'gif_filename', 'training_process.gif')
    fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
    valid_frames = [f for f in visualizer_instance.frame_paths if os.path.exists(f)]
    if not valid_frames:
        return
        
    full_gif_path = os.path.join(visualizer_instance.output_directory, output_filename)
    with imageio.get_writer(full_gif_path, mode='I', fps=fps, loop=0) as writer:
        for f in valid_frames:
            writer.append_data(imageio.imread(f))

    try:
        if os.path.exists(full_gif_path):
            print("[*] Displaying training progress GIF on Colab...")
            display(Image(filename=full_gif_path))
    except Exception as e:
        print(f"[!] Unable to display GIF directly: {e}")