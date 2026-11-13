def execute_training_process(pinn_instance):
    if pinn_instance.pinn_architecture_instance is None:
        pinn_instance._build_system()

    def resample_data_fn():
        return pinn_instance.collocation_sampler_instance.generate_all_domain_points()

    domains_data = resample_data_fn()
    t_cfg = pinn_instance.training_config
    tm = pinn_instance.training_manager_instance

    pre_ep = getattr(t_cfg, 'pretrain_nu_epochs', 300)
    pre_lr = getattr(t_cfg, 'pretrain_nu_lr', 1e-2)
    tm.pretrain_nu_networks(domains_data, epochs=pre_ep, lr=pre_lr)

    total_adam_epochs = t_cfg.number_of_adam_epochs
    warmup_ratio = float(getattr(t_cfg, 'linear_warmup_epochs_ratio', 0.3))
    epochs_warmup = int(total_adam_epochs * warmup_ratio) if getattr(t_cfg, 'enable_linear_warmup_stage', True) else 0
    epochs_adaptation = (total_adam_epochs - epochs_warmup) if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) else 0

    if getattr(t_cfg, 'enable_linear_warmup_stage', True) and epochs_warmup > 0:
        tm.current_fpi_step = 0
        print(f"\n--- [STAGE 1A] Adam Linear Warm-up ({epochs_warmup} Epochs) ---")
        if getattr(tm, 'relobralo_balancer', None) is not None:
            tm.relobralo_balancer.reset_anchors(reset_weights=False)
        domains_data = tm.train_adam(
            epochs=epochs_warmup, domains_data=domains_data,
            dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn, resample_interval=pinn_instance.sampler_config.resample_interval
        )

        if getattr(t_cfg, 'enable_lbfgs_after_warmup', False):
            lbfgs_warmup_iters = int(getattr(t_cfg, 'lbfgs_warmup_iterations', getattr(t_cfg, 'number_of_lbfgs_iterations', 1500)))
            if lbfgs_warmup_iters > 0:
                print(f"\n--- [STAGE 1A-LBFGS] L-BFGS Warm-up Refinement ({lbfgs_warmup_iters} Iterations) ---")
                domains_data = tm.train_lbfgs(max_iterations=lbfgs_warmup_iters, domains_data=domains_data)

    if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) and epochs_adaptation > 0:
        fpi_iters = int(getattr(t_cfg, 'fpi_outer_iterations', 10))
        alpha_damp = float(getattr(t_cfg, 'fpi_alpha_damping', 1.0))
        fpi_tol = float(getattr(t_cfg, 'fpi_tolerance', 1e-5))
        epochs_per_fpi = max(1, epochs_adaptation // fpi_iters)

        fit_ep = getattr(t_cfg, 'fit_nu_epochs', 250)
        fit_lr = getattr(t_cfg, 'fit_nu_lr', 3e-3)

        print(f"\n--- [STAGE 1B] FPI Outer Loop ({fpi_iters} iterations, {epochs_per_fpi} epochs/iter) ---")
        for f_iter in range(fpi_iters):
            tm.current_fpi_step = f_iter + 1
            print(f"\n>>> [FPI OUTER LOOP {f_iter + 1}/{fpi_iters}] <<<")
            if getattr(tm, 'relobralo_balancer', None) is not None:
                tm.relobralo_balancer.reset_anchors(reset_weights=False)
            domains_data = tm.train_adam(
                epochs=epochs_per_fpi, domains_data=domains_data,
                dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn, resample_interval=pinn_instance.sampler_config.resample_interval
            )
            max_diff = tm.fit_nu_network_step(
                domains_data, alpha_damping=alpha_damp, epochs=fit_ep, lr=fit_lr
            )
            print(f"[*] Outer FPI Iter {f_iter + 1}: Max Nu Diff = {max_diff:.4e}")

            if getattr(t_cfg, 'enable_lbfgs_after_fpi_step', False):
                lbfgs_fpi_iters = int(getattr(t_cfg, 'lbfgs_fpi_step_iterations', getattr(t_cfg, 'number_of_lbfgs_iterations', 1500)))
                if lbfgs_fpi_iters > 0:
                    print(f"\n--- [STAGE 1B-LBFGS] L-BFGS FPI Step Refinement ({lbfgs_fpi_iters} Iterations) ---")
                    domains_data = tm.train_lbfgs(max_iterations=lbfgs_fpi_iters, domains_data=domains_data)

            if max_diff < fpi_tol:
                print(f"[OK] FPI Early convergence at iteration {f_iter + 1}!")
                break

    if getattr(t_cfg, 'enable_lbfgs_refinement_stage', True) and t_cfg.number_of_lbfgs_iterations > 0:
        if getattr(pinn_instance.sampler_config, 'resample_points_before_lbfgs_stage', True):
            domains_data = resample_data_fn()

        lbfgs_iters = int(t_cfg.number_of_lbfgs_iterations)
        print(f"\n--- [STAGE 2] L-BFGS Refinement ({lbfgs_iters} Iterations) ---")
        tm.train_lbfgs(max_iterations=lbfgs_iters, domains_data=domains_data)

    if pinn_instance.visualizer_instance:
        pinn_instance.visualizer_instance.generate_animations()