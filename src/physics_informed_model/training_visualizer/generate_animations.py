def generate_animations(visualizer_instance):
    cfg = visualizer_instance.electro_magnetic_pinn.visualization_config
    if getattr(cfg, 'generate_gif_animation', True):
        visualizer_instance.generate_gif()
    if getattr(cfg, 'generate_mp4_video', True):
        visualizer_instance.generate_mp4()
