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
        warmup_lr_begin = getattr(t_cfg, 'adam_learning_rate', 1e-3)
        warmup_lr_finish = getattr(t_cfg, 'adam_eta_min', 1e-6)
        print(f"\n--- [STAGE 1A] Adam Linear Warm-up ({epochs_warmup} Epochs, LR: {warmup_lr_begin} -> {warmup_lr_finish}) ---")
        if getattr(tm, 'relobralo_balancer', None) is not None:
            tm.relobralo_balancer.reset_anchors(reset_weights=False)
        domains_data = tm.train_adam(
            epochs=epochs_warmup,
            domains_data=domains_data,
            dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
            resample_fn=resample_data_fn,
            resample_interval=pinn_instance.sampler_config.resample_interval,
            lr_begin=warmup_lr_begin,
            lr_finish=warmup_lr_finish
        )

        if getattr(t_cfg, 'enable_lbfgs_after_warmup', False):
            lbfgs_warmup_iters = int(getattr(t_cfg, 'lbfgs_warmup_iterations', getattr(t_cfg, 'number_of_lbfgs_iterations', 1500)))
            if lbfgs_warmup_iters > 0:
                print(f"\n--- [STAGE 1A-LBFGS] L-BFGS Warm-up Refinement ({lbfgs_warmup_iters} Iterations) ---")
                domains_data = tm.train_lbfgs(max_iterations=lbfgs_warmup_iters, domains_data=domains_data)

    enable_fpi_l2 = bool(getattr(t_cfg, 'enable_fpi_l2_evaluation', True))

    if getattr(t_cfg, 'enable_nonlinear_adaptation_stage', True) and epochs_adaptation > 0:
        fpi_iters = int(getattr(t_cfg, 'fpi_outer_iterations', 10))
        initial_alpha_damp = float(getattr(t_cfg, 'fpi_initial_alpha_damping', 1.0))
        alpha_damp = float(getattr(t_cfg, 'fpi_alpha_damping', 0.3))
        fpi_tol = float(getattr(t_cfg, 'fpi_tolerance', 1e-5))
        fpi_net_fit_tol = float(getattr(t_cfg, 'fpi_net_fit_tolerance', 1e-2))
        fpi_metric_mode = str(getattr(t_cfg, 'fpi_convergence_metric', 'mu_r')).lower()
        epochs_per_fpi = max(1, epochs_adaptation // fpi_iters)

        fpi_lr_begin = getattr(t_cfg, 'fpi_lr_begin', None)
        if fpi_lr_begin is None:
            fpi_lr_begin = getattr(t_cfg, 'adam_learning_rate', 1e-3)
        fpi_lr_finish = getattr(t_cfg, 'fpi_lr_finish', None)
        if fpi_lr_finish is None:
            fpi_lr_finish = getattr(t_cfg, 'adam_eta_min', 1e-6)

        fit_ep = getattr(t_cfg, 'fit_nu_epochs', 250)
        fit_lr = getattr(t_cfg, 'fit_nu_lr', 3e-3)

        if enable_fpi_l2 and getattr(tm, 'fpi_tracker', None) is not None:
            tm.fpi_tracker.points_per_domain = int(getattr(t_cfg, 'fpi_eval_points_per_domain', 2000))
            if not tm.fpi_tracker.eval_points:
                tm.fpi_tracker.initialize()

        print(f"\n--- [STAGE 1B] FPI Outer Loop ({fpi_iters} iterations, {epochs_per_fpi} epochs/iter, LR: {fpi_lr_begin} -> {fpi_lr_finish}) ---")
        for f_iter in range(fpi_iters):
            tm.current_fpi_step = f_iter + 1
            current_alpha_damp = initial_alpha_damp if f_iter == 0 else alpha_damp
            print(f"\n>>> [FPI OUTER LOOP {f_iter + 1}/{fpi_iters} | damping={current_alpha_damp}] <<<")

            if enable_fpi_l2 and getattr(tm, 'fpi_tracker', None) is not None:
                tm.fpi_tracker.snapshot_before_fpi_step()

            max_diff = tm.fit_nu_network_step(
                domains_data, alpha_damping=current_alpha_damp, epochs=fit_ep, lr=fit_lr
            )

            net_fit_dict = getattr(tm, 'last_nu_fit_metrics', {})
            max_net_fit_mu = max((v.get('net_rel_l2_mu', 0.0) for v in net_fit_dict.values()), default=0.0)
            max_net_fit_nu = max((v.get('net_rel_l2_nu', 0.0) for v in net_fit_dict.values()), default=0.0)

            if enable_fpi_l2 and getattr(tm, 'fpi_tracker', None) is not None and tm.fpi_tracker.eval_points:
                max_fpi_mu, max_fpi_nu, _ = tm.fpi_tracker.evaluate_after_fpi_step(
                    fpi_step=f_iter + 1,
                    net_fit_metrics=net_fit_dict
                )
                print(
                    f"[*] Outer FPI Iter {f_iter + 1} (alpha={current_alpha_damp}): "
                    f"Max Nu Diff = {max_diff:.4e} | Max FPI L2(mu_r) = {max_fpi_mu:.4e} | Max FPI L2(nu*) = {max_fpi_nu:.4e}"
                )
                if fpi_metric_mode == 'nu_star':
                    convergence_metric = max_fpi_nu
                    net_fit_ok = max_net_fit_nu < fpi_net_fit_tol
                elif fpi_metric_mode == 'max_diff':
                    convergence_metric = max_diff
                    net_fit_ok = True
                else:
                    convergence_metric = max_fpi_mu
                    net_fit_ok = max_net_fit_mu < fpi_net_fit_tol
            else:
                print(f"[*] Outer FPI Iter {f_iter + 1} (alpha={current_alpha_damp}): Max Nu Diff = {max_diff:.4e}")
                convergence_metric = max_diff
                net_fit_ok = True

            if f_iter > 0 and convergence_metric < fpi_tol and net_fit_ok:
                print(
                    f"[OK] FPI Early convergence at iteration {f_iter + 1} "
                    f"(metric={convergence_metric:.4e} < {fpi_tol:.4e})!"
                )
                break

            if getattr(tm, 'relobralo_balancer', None) is not None:
                tm.relobralo_balancer.reset_anchors(reset_weights=False)
            domains_data = tm.train_adam(
                epochs=epochs_per_fpi,
                domains_data=domains_data,
                dynamic_resampling=pinn_instance.sampler_config.dynamic_resampling,
                resample_fn=resample_data_fn,
                resample_interval=pinn_instance.sampler_config.resample_interval,
                lr_begin=fpi_lr_begin,
                lr_finish=fpi_lr_finish
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

    if enable_fpi_l2 and getattr(tm, 'fpi_tracker', None) is not None:
        tm.fpi_tracker.evaluate_against_bh_reference()

    if pinn_instance.visualizer_instance:
        pinn_instance.visualizer_instance.generate_animations()