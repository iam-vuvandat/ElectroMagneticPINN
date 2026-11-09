import os

# Đường dẫn đến file scenarios/excute2.py
excute2_path = os.path.join("scenarios", "excute2.py")

if os.path.exists(excute2_path):
    with open(excute2_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Giảm số bước FPI còn 4
    content = content.replace("fpi_iterations = 5", "fpi_iterations = 4")

    # 2. Giảm nhẹ số điểm collocation
    replacements_points = {
        "iron_yoke.training_config.number_of_interior_points = 15000": "iron_yoke.training_config.number_of_interior_points = 6000",
        "iron_yoke.training_config.number_of_surface_points = 4000": "iron_yoke.training_config.number_of_surface_points = 1500",
        "magnet_left.training_config.number_of_interior_points = 1500": "magnet_left.training_config.number_of_interior_points = 800",
        "magnet_left.training_config.number_of_surface_points = 800": "magnet_left.training_config.number_of_surface_points = 400",
        "magnet_right.training_config.number_of_interior_points = 1500": "magnet_right.training_config.number_of_interior_points = 800",
        "magnet_right.training_config.number_of_surface_points = 800": "magnet_right.training_config.number_of_surface_points = 400",
        "number_of_interior_points=6000": "number_of_interior_points=2500"
    }
    for old_str, new_str in replacements_points.items():
        content = content.replace(old_str, new_str)

    # 3. Giảm nhẹ kích thước mạng neural (layers, neurons, fourier features)
    replacements_net = {
        # Mạng iron_yoke
        "iron_yoke.network_config.hidden_layers = 4": "iron_yoke.network_config.hidden_layers = 3",
        "iron_yoke.network_config.hidden_neurons = 128": "iron_yoke.network_config.hidden_neurons = 64",
        "iron_yoke.network_config.fourier_mapping_size = 128": "iron_yoke.network_config.fourier_mapping_size = 64",
        
        # Mạng nu_network của iron_yoke
        "iron_yoke.nu_network_config.hidden_layers = 3": "iron_yoke.nu_network_config.hidden_layers = 2",
        "iron_yoke.nu_network_config.hidden_neurons = 64": "iron_yoke.nu_network_config.hidden_neurons = 32",
        "iron_yoke.nu_network_config.fourier_mapping_size = 32": "iron_yoke.nu_network_config.fourier_mapping_size = 16",
        
        # Mạng magnet_left
        "magnet_left.network_config.hidden_layers = 4": "magnet_left.network_config.hidden_layers = 3",
        "magnet_left.network_config.hidden_neurons = 64": "magnet_left.network_config.hidden_neurons = 32",
        "magnet_left.network_config.fourier_mapping_size = 64": "magnet_left.network_config.fourier_mapping_size = 32",
        
        # Mạng magnet_right
        "magnet_right.network_config.hidden_layers = 4": "magnet_right.network_config.hidden_layers = 3",
        "magnet_right.network_config.hidden_neurons = 64": "magnet_right.network_config.hidden_neurons = 32",
        "magnet_right.network_config.fourier_mapping_size = 64": "magnet_right.network_config.fourier_mapping_size = 32",
        
        # Mạng background
        "hidden_layers=4,\n    hidden_neurons=64,": "hidden_layers=3,\n    hidden_neurons=32,",
        "fourier_mapping_size=64\n)": "fourier_mapping_size=32\n)"
    }
    for old_str, new_str in replacements_net.items():
        content = content.replace(old_str, new_str)

    with open(excute2_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully tuned network parameters and points in {excute2_path}")
else:
    print(f"File not found: {excute2_path}")