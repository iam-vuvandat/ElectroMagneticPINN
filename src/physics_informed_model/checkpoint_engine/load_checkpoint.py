import io
import pickle
from pathlib import Path
import torch


class _DeviceUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location="cpu"):
        super().__init__(file)
        self.map_location = map_location

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        return super().find_class(module, name)


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
        pkl_candidate = Path(f"{path}.pkl")
        if pkl_candidate.exists():
            path = pkl_candidate
        elif path.suffix == ".pkl" and path.with_suffix("").exists():
            path = path.with_suffix("")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(path, "rb") as f:
        checkpoint_data = _DeviceUnpickler(f, map_location=target_device).load()

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
    else:
        pinn_instance.computation_device = target_device
        pinn_instance.pinn_architecture_instance.to(target_device)

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
        if (
            checkpoint_data.get("relobralo_state_dict")
            and getattr(tm, "relobralo_balancer", None)
        ):
            tm.relobralo_balancer.load_state_dict(
                checkpoint_data["relobralo_state_dict"]
            )
        tm.completed_adam_epochs = checkpoint_data.get(
            "completed_adam_epochs", 0
        )
        tm.global_step = checkpoint_data.get("global_step", 0)
        tm.current_fpi_step = checkpoint_data.get("current_fpi_step", 0)

    vis = pinn_instance.visualizer_instance
    if vis:
        vis.loss_history = checkpoint_data.get("loss_history", [])
        vis.epoch_history = checkpoint_data.get("epoch_history", [])
        vis.pde_loss_history = checkpoint_data.get("pde_loss_history", [])
        vis.az_loss_history = checkpoint_data.get("az_loss_history", [])
        vis.bn_loss_history = checkpoint_data.get("bn_loss_history", [])
        vis.ht_loss_history = checkpoint_data.get("ht_loss_history", [])
        vis.w_az_history = checkpoint_data.get("w_az_history", [])
        vis.w_bn_history = checkpoint_data.get("w_bn_history", [])
        vis.w_ht_history = checkpoint_data.get("w_ht_history", [])

    print(
        f"[*] Successfully loaded single-file pickle checkpoint from: {path}"
    )
    return pinn_instance