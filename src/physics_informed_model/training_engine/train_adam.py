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

    dynamic_interval = getattr(t_cfg, 'dynamic_weight_update_interval', 50)
    enable_dynamic = getattr(t_cfg, 'enable_dynamic_weighting', False)
    dynamic_method = getattr(t_cfg, 'dynamic_weight_method', 'grad_norm')

    for param_group in training_manager_instance.optimizer_adam.param_groups:
        param_group['initial_lr'] = training_manager_instance.base_learning_rate_adam
        param_group['lr'] = training_manager_instance.base_learning_rate_adam
        
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(training_manager_instance.optimizer_adam, T_max=max(epochs, 1), eta_min=eta_min)
    stop_reason = f"Hoàn thành toàn bộ {epochs} epochs."
    non_finite_streak = 0
    training_manager_instance.completed_adam_epochs = 0
    
    for epoch in range(epochs):
        if dynamic_resampling and resample_fn and epoch > 0 and epoch % resample_interval == 0:
            domains_data = resample_fn()

        # Tự động cập nhật trọng số động theo Grad-Norm Balancing theo chu kỳ
        if enable_dynamic and dynamic_method == "grad_norm" and epoch > 0 and epoch % dynamic_interval == 0:
            training_manager_instance.update_dynamic_loss_weights()

        training_manager_instance.optimizer_adam.zero_grad(set_to_none=True)
        loss = training_manager_instance.compute_loss(domains_data)
        
        if not torch.isfinite(loss):
            non_finite_streak += 1
            if non_finite_streak >= max_non_finite_streak:
                stop_reason = f"Dừng sớm do Loss không hữu hạn (NaN/Inf) liên tiếp {max_non_finite_streak} lần."
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
            print(
                f"Adam Epoch {epoch + 1:05d} | Total = {current_loss_value:.4e} | "
                f"PDE = {bd.get('pde_loss', 0.0):.3e} | "
                f"IF(Az/Bn/Ht) = {bd.get('interface_Az', 0.0):.2e}/{bd.get('interface_Bn', 0.0):.2e}/{bd.get('interface_Ht', 0.0):.2e}"
            )
            
        if training_manager_instance.visualizer and (epoch + 1) % training_manager_instance.visualizer_update_interval == 0:
            training_manager_instance.visualizer.save_frame(epoch + 1, current_loss_value, domains_data)
            
    print(f"[*] KẾT THÚC ADAM. Lý do: {stop_reason}")
    return domains_data
