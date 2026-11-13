import pickle
from pathlib import Path
from types import SimpleNamespace
import torch


def get_project_root():
    current_file = Path(__file__).resolve()
    return current_file.parents[3]


def resolve_checkpoint_filepath(filepath=None):
    root_dir = get_project_root()
    default_dir = root_dir / "checkpoints"

    if filepath is None:
        return default_dir / "pinn_model_checkpoint.pkl"

    path_obj = Path(filepath)

    if path_obj.is_absolute():
        return path_obj

    if path_obj.parent != Path("."):
        return (root_dir / path_obj).resolve()

    return default_dir / path_obj.name


def _move_tensors_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_cpu(v) for v in obj)
    if isinstance(obj, SimpleNamespace):
        return SimpleNamespace(**{k: _move_tensors_to_cpu(v) for k, v in vars(obj).items()})
    return obj


def save_checkpoint(pinn_instance, filepath=None):
    path = resolve_checkpoint_filepath(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arch = pinn_instance.pinn_architecture_instance
    tm = pinn_instance.training_manager_instance
    sim = pinn_instance.simulation_domain_engine_instance
    vis = pinn_instance.visualizer_instance

    checkpoint_data = {
        "model_state_dict": arch.state_dict() if arch else None,
        "optimizer_adam_state_dict": (
            tm.optimizer_adam.state_dict() if tm and tm.optimizer_adam else None
        ),
        "optimizer_lbfgs_state_dict": (
            tm.optimizer_lbfgs.state_dict()
            if tm and tm.optimizer_lbfgs
            else None
        ),
        "relobralo_state_dict": (
            tm.relobralo_balancer.state_dict()
            if tm and getattr(tm, "relobralo_balancer", None)
            else None
        ),
        "completed_adam_epochs": getattr(tm, "completed_adam_epochs", 0) if tm else 0,
        "global_step": getattr(tm, "global_step", 0) if tm else 0,
        "current_fpi_step": getattr(tm, "current_fpi_step", 0) if tm else 0,
        "sampler_config": pinn_instance.sampler_config,
        "training_config": pinn_instance.training_config,
        "visualization_config": pinn_instance.visualization_config,
        "physics_config": sim.physics_config if sim else None,
        "x_boundaries_tuple": sim.x_boundaries_tuple if sim else None,
        "y_boundaries_tuple": sim.y_boundaries_tuple if sim else None,
        "loss_history": vis.loss_history if vis else [],
        "epoch_history": vis.epoch_history if vis else [],
        "pde_loss_history": getattr(vis, "pde_loss_history", []) if vis else [],
        "az_loss_history": getattr(vis, "az_loss_history", []) if vis else [],
        "bn_loss_history": getattr(vis, "bn_loss_history", []) if vis else [],
        "ht_loss_history": getattr(vis, "ht_loss_history", []) if vis else [],
        "w_az_history": getattr(vis, "w_az_history", []) if vis else [],
        "w_bn_history": getattr(vis, "w_bn_history", []) if vis else [],
        "w_ht_history": getattr(vis, "w_ht_history", []) if vis else [],
    }

    checkpoint_data = _move_tensors_to_cpu(checkpoint_data)

    with open(path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"[*] Successfully saved single-file pickle checkpoint to: {path}")