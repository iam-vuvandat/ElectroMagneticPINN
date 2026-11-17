import time
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
    last_loss_val = [None]
    last_saved_counter = [0]

    def closure():
        iter_start = time.perf_counter()
        training_manager_instance.optimizer_lbfgs.zero_grad(set_to_none=True)
        loss = training_manager_instance.compute_loss(
            domains_data,
            update_relobralo=False,
            is_resampling_step=False
        )
        if not torch.isfinite(loss):
            return loss
        loss.backward()

        iter_elapsed = time.perf_counter() - iter_start
        lbfgs_counter[0] += 1
        training_manager_instance.global_step += 1
        current_loss_value = loss.item()
        last_loss_val[0] = current_loss_value

        if training_manager_instance.visualizer:
            training_manager_instance.visualizer.record_history(
                training_manager_instance.global_step,
                current_loss_value,
                loss_breakdown=training_manager_instance.last_loss_breakdown
            )

        if lbfgs_counter[0] == 1 or lbfgs_counter[0] % log_interval == 0:
            bd = training_manager_instance.last_loss_breakdown
            lmb = bd.get('relobralo_lambdas', {})
            bc_str = f" | BC = {bd.get('boundary_loss', 0.0):.2e}" if 'loss_bc' in lmb else ""
            print(
                f"L-BFGS Closure {lbfgs_counter[0]:05d} | Total = {current_loss_value:.4e} | "
                f"PDE = {bd.get('pde_loss', 0.0):.3e}{bc_str} | "
                f"Time = {iter_elapsed * 1000.0:.2f}ms"
            )

        if training_manager_instance.visualizer and (
            lbfgs_counter[0] % training_manager_instance.visualizer_update_interval == 0
        ):
            last_saved_counter[0] = lbfgs_counter[0]
            training_manager_instance.visualizer.save_frame(
                training_manager_instance.global_step,
                current_loss_value,
                domains_data,
                loss_breakdown=training_manager_instance.last_loss_breakdown,
                fpi_step=getattr(training_manager_instance, 'current_fpi_step', 0)
            )

        return loss

    training_manager_instance.optimizer_lbfgs.step(closure)

    if (
        training_manager_instance.visualizer
        and last_loss_val[0] is not None
        and lbfgs_counter[0] > 0
        and last_saved_counter[0] != lbfgs_counter[0]
    ):
        training_manager_instance.visualizer.save_frame(
            training_manager_instance.global_step,
            last_loss_val[0],
            domains_data,
            loss_breakdown=training_manager_instance.last_loss_breakdown,
            fpi_step=getattr(training_manager_instance, 'current_fpi_step', 0)
        )

    return domains_data