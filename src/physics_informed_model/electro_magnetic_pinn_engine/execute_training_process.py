import torch.optim as optim


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
    default_min_epochs = int(getattr(t_cfg, 'min_adam_epochs', 3000))
    warmup_min_epochs = int(getattr(t_cfg, 'warmup_min_epochs', default_min_epochs))
    fpi_min_epochs = int(getattr(t_cfg, 'fpi_min_epochs', default_min_epochs))
    warmup_max_cfg = getattr(t_cfg, 'warmup_max_epochs', None)
    fpi_max_cfg = getattr(t_cfg, 'fpi_max_epochs', None)

    warmup_ratio = float(getattr(t_cfg, 'linear_warmup_epochs_ratio', 0.3))
    enable_warmup = bool(getattr(t_cfg, 'enable_linear_warmup_stage', True))
    enable_adaptation = bool(getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True))

    if enable_warmup:
        if warmup_max_cfg is not None:
            epochs_warmup = max(int(warmup_max_cfg), warmup_min_epochs)
        else:
            epochs_warmup = max(int(total_adam_epochs * warmup_ratio), warmup_min_epochs)
    else:
        epochs_warmup = 0

    fpi_iters = int(getattr(t_cfg, 'fpi_outer_iterations', 16))
    if enable_adaptation:
        if fpi_max_cfg is not None:
            epochs_per_fpi = max(int(fpi_max_cfg), fpi_min_epochs)
        else:
            epochs_adaptation = max(0, total_adam_epochs - epochs_warmup)
            epochs_per_fpi = max(fpi_min_epochs, epochs_adaptation // max(fpi_iters, 1)) if epochs_adaptation > 0 else 0
    else:
        epochs_per_fpi = 0

    total_fpi_epochs = fpi_iters * epochs_per_fpi
    eta_min = float(getattr(t_cfg, 'adam_eta_min', 1e-5))
    scheduler_type = str(getattr(t_cfg, 'lr_scheduler_type', 'linear')).lower()
    custom_fpi_lr = getattr(t_cfg, 'fpi_adam_learning_rate', None)

    def build_adam_scheduler(optimizer, base_lr, total_iters):
        steps = max(int(total_iters), 1)
        if scheduler_type == 'linear':
            end_factor = min(max(eta_min / max(base_lr, 1e-12), 1e-8), 1.0)
            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=steps
            )
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=eta_min
        )

    shared_scheduler = None
    if custom_fpi_lr is None:
        for param_group in tm.optimizer_adam.param_groups:
            param_group['initial_lr'] = tm.base_learning_rate_adam
            param_group['lr'] = tm.base_learning_rate_adam
        total_schedule_epochs = max(epochs_warmup + total_fpi_epochs, 1)
        shared_scheduler = build_adam_scheduler(
            tm.optimizer_adam,
            tm.base_learning_rate_adam,
            total_schedule_epochs
        )

    if enable_warmup and epochs_warmup > 0:
        tm.current_fpi_step = 0
        print(f"\n--- [STAGE 1A] Adam Linear Warm-up (min={warmup_min_epochs}, max={epochs_warmup} Epochs) ---")
        if getattr(tm, 'relobralo_balancer', None) is not None:
            tm.relobralo_balancer.reset_anchors(reset_weights=False)
        domains_data = tm.train_adam(
            epochs=epochs_warmup,
            domains_data=domains_data,
            dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn,
            resample_interval=pinn_instance.sampler_config.resample_interval,
            reset_lr=(shared_scheduler is None),
            scheduler=shared_scheduler,
            min_epochs=warmup_min_epochs
        )

        if getattr(t_cfg, 'enable_lbfgs_after_warmup', False):
            lbfgs_warmup_iters = int(getattr(t_cfg, 'lbfgs_warmup_iterations', getattr(t_cfg, 'number_of_lbfgs_iterations', 1500)))
            if lbfgs_warmup_iters > 0:
                print(f"\n--- [STAGE 1A-LBFGS] L-BFGS Warm-up Refinement ({lbfgs_warmup_iters} Iterations) ---")
                domains_data = tm.train_lbfgs(max_iterations=lbfgs_warmup_iters, domains_data=domains_data)

    if enable_adaptation and epochs_per_fpi > 0:
        initial_alpha_damp = float(getattr(t_cfg, 'fpi_initial_alpha_damping', 1.0))
        alpha_damp = float(getattr(t_cfg, 'fpi_alpha_damping', 0.3))
        fpi_tol = float(getattr(t_cfg, 'fpi_tolerance', 1e-5))

        fit_ep = getattr(t_cfg, 'fit_nu_epochs', 250)
        fit_lr = getattr(t_cfg, 'fit_nu_lr', 3e-3)

        if shared_scheduler is not None:
            fpi_scheduler = shared_scheduler
        else:
            fpi_lr_start = float(custom_fpi_lr)
            for param_group in tm.optimizer_adam.param_groups:
                param_group['initial_lr'] = fpi_lr_start
                param_group['lr'] = fpi_lr_start

            fpi_scheduler = build_adam_scheduler(
                tm.optimizer_adam,
                fpi_lr_start,
                max(total_fpi_epochs, 1)
            )

        print(f"\n--- [STAGE 1B] FPI Outer Loop ({fpi_iters} iterations, min={fpi_min_epochs}, max={epochs_per_fpi} epochs/iter, continuous LR) ---")
        for f_iter in range(fpi_iters):
            tm.current_fpi_step = f_iter + 1
            if getattr(pinn_instance.sampler_config, 'resample_points_before_fpi_step', True):
                domains_data = resample_data_fn()

            current_alpha_damp = initial_alpha_damp if f_iter == 0 else alpha_damp
            current_lr = tm.optimizer_adam.param_groups[0]['lr']
            print(f"\n>>> [FPI OUTER LOOP {f_iter + 1}/{fpi_iters} | damping={current_alpha_damp} | lr={current_lr:.3e}] <<<")

            max_diff = tm.fit_nu_network_step(
                domains_data, alpha_damping=current_alpha_damp, epochs=fit_ep, lr=fit_lr
            )
            print(f"[*] Outer FPI Iter {f_iter + 1} (alpha={current_alpha_damp}): Max Nu Diff = {max_diff:.4e}")

            if f_iter > 0 and max_diff < fpi_tol:
                print(f"[OK] FPI Early convergence at iteration {f_iter + 1}!")
                break

            if getattr(tm, 'relobralo_balancer', None) is not None:
                tm.relobralo_balancer.reset_anchors(reset_weights=False)
            domains_data = tm.train_adam(
                epochs=epochs_per_fpi,
                domains_data=domains_data,
                dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn,
                resample_interval=pinn_instance.sampler_config.resample_interval,
                reset_lr=False,
                scheduler=fpi_scheduler,
                min_epochs=fpi_min_epochs
            )

            if getattr(t_cfg, 'enable_lbfgs_after_fpi_step', False):
                lbfgs_fpi_iters = int(getattr(t_cfg, 'lbfgs_fpi_step_iterations', getattr(t_cfg, 'number_of_lbfgs_iterations', 1500)))
                if lbfgs_fpi_iters > 0:
                    print(f"\n--- [STAGE 1B-LBFGS] L-BFGS FPI Step Refinement ({lbfgs_fpi_iters} Iterations) ---")
                    domains_data = tm.train_lbfgs(max_iterations=lbfgs_fpi_iters, domains_data=domains_data)

    if getattr(t_cfg, 'enable_lbfgs_refinement_stage', True) and t_cfg.number_of_lbfgs_iterations > 0:
        if getattr(pinn_instance.sampler_config, 'resample_points_before_lbfgs_stage', True):
            domains_data = resample_data_fn()

        lbfgs_iters = int(t_cfg.number_of_lbfgs_iterations)
        print(f"\n--- [STAGE 2] L-BFGS Refinement ({lbfgs_iters} Iterations) ---")
        tm.train_lbfgs(max_iterations=lbfgs_iters, domains_data=domains_data)

    if pinn_instance.visualizer_instance:
        pinn_instance.visualizer_instance.generate_animations()