import pickle
from pathlib import Path


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


def save_checkpoint(pinn_instance, filepath=None):
    path = resolve_checkpoint_filepath(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arch = pinn_instance.pinn_architecture_instance
    tm = pinn_instance.training_manager_instance
    sim = pinn_instance.simulation_domain_engine_instance

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
        "completed_adam_epochs": getattr(tm, "completed_adam_epochs", 0)
        if tm
        else 0,
        "global_step": getattr(tm, "global_step", 0) if tm else 0,
        "sampler_config": pinn_instance.sampler_config,
        "training_config": pinn_instance.training_config,
        "visualization_config": pinn_instance.visualization_config,
        "physics_config": sim.physics_config if sim else None,
        "x_boundaries_tuple": sim.x_boundaries_tuple if sim else None,
        "y_boundaries_tuple": sim.y_boundaries_tuple if sim else None,
        "loss_history": (
            pinn_instance.visualizer_instance.loss_history
            if pinn_instance.visualizer_instance
            else []
        ),
        "epoch_history": (
            pinn_instance.visualizer_instance.epoch_history
            if pinn_instance.visualizer_instance
            else []
        ),
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"[*] Successfully saved single-file pickle checkpoint to: {path}")
