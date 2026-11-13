import torch
import torch.optim as optim


def train_adam(training_manager_instance, epochs, domains_data, dynamic_resampling=False, resample_fn=None, resample_interval=100):
    model = training_manager_instance.model
    model.train()
    pinn = training_manager_instance.electro_magnetic_pinn
    t_cfg = getattr(pinn, 'training_config', None)

    eta_min = getattr(t_cfg, 'adam_eta_min', 1e-6)
    max_clip_norm = getattr(t_cfg, 'grad_clip_max_norm', 1.0)
    max_non_finite_streak = getattr(t_cfg, 'max_non_finite_streak', 5)
    log_interval = getattr(t_cfg, 'adam_log_interval', 100)

    for param_group in training_manager_instance.optimizer_adam.param_groups:
        param_group['initial_lr'] = training_manager_instance.base_learning_rate_adam
        param_group['lr'] = training_manager_instance.base_learning_rate_adam
        
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(training_manager_instance.optimizer_adam, T_max=max(epochs, 1), eta_min=eta_min)
    stop_reason = f"Completed all {epochs} epochs."
    non_finite_streak = 0
    training_manager_instance.completed_adam_epochs = 0
    
    for epoch in range(epochs):
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
        scheduler_adam.step()
        
        training_manager_instance.completed_adam_epochs = epoch + 1
        current_loss_value = loss.item()
        
        if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0:
            bd = training_manager_instance.last_loss_breakdown
            if getattr(training_manager_instance, 'relobralo_balancer', None) is not None:
                lmb = bd.get('relobralo_lambdas', {})
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e} | "
                    f"Lambda(pde/a/bn/ht) = {lmb.get('loss_pde', 1.0):.2f}/{lmb.get('loss_a', 1.0):.2f}/{lmb.get('loss_bn', 1.0):.2f}/{lmb.get('loss_ht', 1.0):.2f}"
                )
            else:
                print(
                    f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                    f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                    f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
                )
            
        if training_manager_instance.visualizer and (epoch + 1) % training_manager_instance.visualizer_update_interval == 0:
            training_manager_instance.visualizer.save_frame(
                training_manager_instance.global_step,
                current_loss_value,
                domains_data,
                loss_breakdown=training_manager_instance.last_loss_breakdown,
                fpi_step=getattr(training_manager_instance, 'current_fpi_step', 0)
            )
            
    print(f"[*] ADAM FINISHED. Reason: {stop_reason}")
    return domains_data