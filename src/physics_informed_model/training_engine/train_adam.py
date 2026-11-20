import time
import torch
import torch.optim as optim


def train_adam(training_manager_instance, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
    model = training_manager_instance.model
    model.train()
    pinn = training_manager_instance.electro_magnetic_pinn
    t_cfg = getattr(pinn, 'training_config', None)

    base_lr = float(getattr(t_cfg, 'adam_learning_rate', training_manager_instance.base_learning_rate_adam))
    eta_min = float(getattr(t_cfg, 'adam_eta_min', 1e-6))
    scheduler_mode = str(getattr(t_cfg, 'adam_lr_scheduler', getattr(training_manager_instance, 'adam_lr_scheduler', 'linear'))).lower()
    max_clip_norm = getattr(t_cfg, 'grad_clip_max_norm', 1.0)
    max_non_finite_streak = getattr(t_cfg, 'max_non_finite_streak', 5)
    log_interval = getattr(t_cfg, 'adam_log_interval', 100)
    target_loss_threshold = getattr(
        t_cfg,
        'target_loss_threshold',
        getattr(training_manager_instance, 'target_loss_threshold', None)
    )

    for param_group in training_manager_instance.optimizer_adam.param_groups:
        param_group['initial_lr'] = base_lr
        param_group['lr'] = base_lr

    total_iters = max(int(epochs), 1)
    if scheduler_mode == 'linear':
        end_factor = min(max(eta_min / max(base_lr, 1e-12), 0.0), 1.0)
        scheduler_adam = optim.lr_scheduler.LinearLR(
            training_manager_instance.optimizer_adam,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=total_iters
        )
    elif scheduler_mode == 'cosine':
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(
            training_manager_instance.optimizer_adam,
            T_max=total_iters,
            eta_min=eta_min
        )
    else:
        scheduler_adam = None

    stop_reason = f"Completed all {epochs} epochs."
    non_finite_streak = 0
    training_manager_instance.completed_adam_epochs = 0

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        training_manager_instance.global_step += 1
        is_resampling_step = False
        if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
            domains_data = resample_fn()
            is_resampling_step = True

        training_manager_instance.optimizer_adam.zero_grad(set_to_none=True)
        loss = training_manager_instance.compute_loss(
            domains_data,
            update_relobralo=True,
            is_resampling_step=is_resampling_step
        )

        if not torch.isfinite(loss):
            non_finite_streak += 1
            if non_finite_streak >= max_non_finite_streak:
                stop_reason = f"Early stop due to non-finite Loss (NaN/Inf) for {max_non_finite_streak} consecutive steps."
                break
            continue
        non_finite_streak = 0

        loss.backward()
        if max_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.networks.parameters(), max_norm=max_clip_norm)
        training_manager_instance.optimizer_adam.step()
        if scheduler_adam is not None:
            scheduler_adam.step()

        epoch_elapsed = time.perf_counter() - epoch_start_time

        training_manager_instance.completed_adam_epochs = epoch + 1
        current_loss_value = loss.item()

        if training_manager_instance.visualizer:
            training_manager_instance.visualizer.record_history(
                training_manager_instance.global_step,
                current_loss_value,
                loss_breakdown=training_manager_instance.last_loss_breakdown
            )

        reached_target = (
            target_loss_threshold is not None
            and current_loss_value <= float(target_loss_threshold)
        )

        if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0 or reached_target:
            bd = training_manager_instance.last_loss_breakdown
            bc_val = bd.get('boundary_loss', 0.0)
            current_lr = training_manager_instance.optimizer_adam.param_groups[0]['lr']
            if getattr(training_manager_instance, 'relobralo_balancer', None) is not None:
                lmb = bd.get('relobralo_lambdas', {})
                has_bc = 'loss_bc' in lmb
                bc_str = f" | BC = {bc_val:.2e}" if has_bc else ""
                lmb_keys_label = "Lambda(pde/a/bn/ht/bc)" if has_bc else "Lambda(pde/a/bn/ht)"
                lmb_bc_str = f"/{lmb.get('loss_bc', 1.0):.2f}" if has_bc else ""
                print(
                    f"Adam Epoch {epoch + 1:05d} | LR = {current_lr:.2e} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
                    f"{bc_str} | "
                    f"{lmb_keys_label} = {lmb.get('loss_pde', 1.0):.2f}/{lmb.get('loss_a', 1.0):.2f}/{lmb.get('loss_bn', 1.0):.2f}/{lmb.get('loss_ht', 1.0):.2f}{lmb_bc_str} | "
                    f"Time = {epoch_elapsed * 1000.0:.2f}ms"
                )
            else:
                print(
                    f"Adam Epoch {epoch + 1:05d} | LR = {current_lr:.2e} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e} | "
                    f"BC = {bc_val:.2e} | "
                    f"Time = {epoch_elapsed * 1000.0:.2f}ms"
                )

        if training_manager_instance.visualizer and (
            (epoch + 1) % training_manager_instance.visualizer_update_interval == 0 or reached_target
        ):
            training_manager_instance.visualizer.save_frame(
                training_manager_instance.global_step,
                current_loss_value,
                domains_data,
                loss_breakdown=training_manager_instance.last_loss_breakdown,
                fpi_step=getattr(training_manager_instance, 'current_fpi_step', 0)
            )

        if reached_target:
            stop_reason = f"Early stop at epoch {epoch + 1}: Total loss ({current_loss_value:.4e}) <= target_loss_threshold ({float(target_loss_threshold):.4e})."
            break

    print(f"[*] ADAM FINISHED. Reason: {stop_reason}")
    return domains_data