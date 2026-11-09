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


def load_checkpoint(pinn_instance, filepath=None):
    path = resolve_checkpoint_filepath(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    with open(path, "rb") as f:
        checkpoint_data = pickle.load(f)

    if "sampler_config" in checkpoint_data:
        pinn_instance.sampler_config = checkpoint_data["sampler_config"]
    if "training_config" in checkpoint_data:
        pinn_instance.training_config = checkpoint_data["training_config"]
    if "visualization_config" in checkpoint_data:
        pinn_instance.visualization_config = checkpoint_data[
            "visualization_config"
        ]

    sim = pinn_instance.simulation_domain_engine_instance
    if sim:
        if "physics_config" in checkpoint_data:
            sim.physics_config = checkpoint_data["physics_config"]
        if "x_boundaries_tuple" in checkpoint_data:
            sim.x_boundaries_tuple = checkpoint_data["x_boundaries_tuple"]
        if "y_boundaries_tuple" in checkpoint_data:
            sim.y_boundaries_tuple = checkpoint_data["y_boundaries_tuple"]

    if pinn_instance.pinn_architecture_instance is None:
        pinn_instance._build_system()

    if (
        checkpoint_data.get("model_state_dict")
        and pinn_instance.pinn_architecture_instance
    ):
        pinn_instance.pinn_architecture_instance.load_state_dict(
            checkpoint_data["model_state_dict"]
        )

    tm = pinn_instance.training_manager_instance
    if tm:
        if checkpoint_data.get("optimizer_adam_state_dict") and tm.optimizer_adam:
            tm.optimizer_adam.load_state_dict(
                checkpoint_data["optimizer_adam_state_dict"]
            )
        if (
            checkpoint_data.get("optimizer_lbfgs_state_dict")
            and tm.optimizer_lbfgs
        ):
            tm.optimizer_lbfgs.load_state_dict(
                checkpoint_data["optimizer_lbfgs_state_dict"]
            )
        tm.completed_adam_epochs = checkpoint_data.get(
            "completed_adam_epochs", 0
        )
        tm.global_step = checkpoint_data.get("global_step", 0)

    vis = pinn_instance.visualizer_instance
    if vis:
        vis.loss_history = checkpoint_data.get("loss_history", [])
        vis.epoch_history = checkpoint_data.get("epoch_history", [])

    print(
        f"[*] Successfully loaded single-file pickle checkpoint from: {path}"
    )
    return pinn_instance
