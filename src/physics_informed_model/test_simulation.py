import os

def maximize_fourier_technique():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    test_file = os.path.join(project_root, 'test_simulation.py')
    with open(test_file, 'r', encoding='utf-8') as f:
        test_content = f.read()

    old_config = """    model.pinn_config.hidden_layers = 6
    model.pinn_config.hidden_neurons = 64
    model.pinn_config.activation_function = nn.Tanh()
    
    model.pinn_config.fourier_features = True
    model.pinn_config.fourier_sigma = 0.5
    model.pinn_config.fourier_mapping_size = 256
    
    model.training_config.epochs_adam = 4000
    model.training_config.epochs_lbfgs = 1000
    model.training_config.learning_rate_adam = 1e-3"""

    new_config = """    model.pinn_config.hidden_layers = 6
    model.pinn_config.hidden_neurons = 128
    model.pinn_config.activation_function = nn.Tanh()
    
    model.pinn_config.fourier_features = True
    model.pinn_config.fourier_sigma = 0.85
    model.pinn_config.fourier_mapping_size = 512
    
    model.training_config.epochs_adam = 5000
    model.training_config.epochs_lbfgs = 1000
    model.training_config.learning_rate_adam = 8e-4"""

    if "model.pinn_config.fourier_mapping_size = 256" in test_content:
        test_content = test_content.replace(old_config, new_config)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print("[*] Đã tự động tinh chỉnh siêu tham số tối đa hóa Fourier vào test_simulation.py")
    else:
        print("[!] Không tìm thấy cấu hình cũ để thay thế. Hãy kiểm tra lại tệp test_simulation.py.")

if __name__ == "__main__":
    maximize_fourier_technique()