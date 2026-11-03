import torch
import torch.optim as optim

def train_lbfgs(training_manager_instance, max_iterations, domains_data):
    if max_iterations <= 0:
        return domains_data

    model = training_manager_instance.model
    model.train()
    pinn = training_manager_instance.electro_magnetic_pinn
    t_cfg = getattr(pinn, 'training_config', None)
    log_interval = getattr(t_cfg, 'lbfgs_log_interval', 25)

    training_manager_instance.optimizer_lbfgs = optim.LBFGS(
        model.networks.parameters(),
        lr=training_manager_instance.lbfgs_learning_rate,
        max_iter=max_iterations,
        max_eval=int(max_iterations * 1.25),
        tolerance_grad=training_manager_instance.lbfgs_tolerance_gradient,
        tolerance_change=training_manager_instance.lbfgs_tolerance_change,
        history_size=training_manager_instance.lbfgs_history_size,
        line_search_fn="strong_wolfe"
    )

    lbfgs_counter = [0]
    
    def closure():
        training_manager_instance.optimizer_lbfgs.zero_grad(set_to_none=True)
        loss = training_manager_instance.compute_loss(domains_data)
        if not torch.isfinite(loss):
            return loss
        loss.backward()
        
        lbfgs_counter[0] += 1
        if lbfgs_counter[0] == 1 or lbfgs_counter[0] % log_interval == 0:
            bd = training_manager_instance.last_loss_breakdown
            print(
                f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {loss.item():.4e} | "
                f"PDE = {bd.get('pde_loss', 0.0):.3e}"
            )
        return loss
        
    training_manager_instance.optimizer_lbfgs.step(closure)
    return domains_data
