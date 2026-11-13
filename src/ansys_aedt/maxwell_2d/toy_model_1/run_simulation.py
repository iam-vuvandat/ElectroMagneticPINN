def run_simulation(toy_model_obj):
    print("\033[94mIn function run_simulation.\033[0m")
    print("\033[94m{\033[0m")

    m2d = toy_model_obj.m2d
    if m2d is None:
        raise RuntimeError("Model chưa được khởi tạo! Hãy gọi create_model() trước khi chạy run_simulation().")

    config = toy_model_obj.configuration
    setup_name = config.analysis.setup_name

    print(f"\033[94mIn function run_simulation: Analyzing setup '{setup_name}'...\033[0m")
    m2d.analyze_setup(setup_name)

    print(f"\033[94mIn function run_simulation: Successfully completed simulation for '{config.name}'.\033[0m")
    print("\033[94m}\033[0m\n")
    return toy_model_obj
