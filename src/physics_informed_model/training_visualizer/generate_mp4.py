import os
import imageio

def generate_mp4(visualizer_instance):
    cfg = visualizer_instance.electro_magnetic_pinn.visualization_config
    output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
    fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
    codec = getattr(cfg, 'mp4_codec', 'libx264')
    quality = getattr(cfg, 'mp4_quality', 8)

    valid_frames = [f for f in visualizer_instance.frame_paths if os.path.exists(f)]
    if not valid_frames:
        return

    full_mp4_path = os.path.join(visualizer_instance.output_directory, output_filename)
    try:
        writer = imageio.get_writer(full_mp4_path, fps=fps, codec=codec, pixelformat='yuv420p', quality=quality)
        for f in valid_frames:
            writer.append_data(imageio.imread(f))
        writer.close()
    except Exception as exc:
        pass
