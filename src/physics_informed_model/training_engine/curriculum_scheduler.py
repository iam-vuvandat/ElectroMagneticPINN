import math


class CurriculumWeightScheduler:
    def __init__(
        self,
        total_warmup_epochs,
        stage_order=("loss_ht", "loss_a", "loss_bn", "loss_pde"),
        steepness=15.0,
        min_weights=None,
        max_weights=None,
        couple_bc_with_az=True
    ):
        self.total_warmup_epochs = max(int(total_warmup_epochs), 0)
        self.stage_order = tuple(stage_order)
        self.steepness = float(steepness)
        self.couple_bc_with_az = bool(couple_bc_with_az)

        default_min = {
            "loss_ht": 1.0,
            "loss_a": 0.02,
            "loss_bc": 0.05,
            "loss_bn": 0.01,
            "loss_pde": 1e-3
        }
        default_max = {k: 1.0 for k in default_min}

        if min_weights:
            default_min.update({k: float(v) for k, v in min_weights.items()})
        if max_weights:
            default_max.update({k: float(v) for k, v in max_weights.items()})

        self.min_weights = default_min
        self.max_weights = default_max

        num_stages = max(len(self.stage_order), 1)
        self.stage_centers = {
            key: float(idx) / float(num_stages)
            for idx, key in enumerate(self.stage_order)
        }

    def is_active(self, step):
        return self.total_warmup_epochs > 0 and int(step) < self.total_warmup_epochs

    def _smooth_gate(self, progress, center, min_w, max_w):
        if center <= 0.0:
            return float(max_w)
        exponent = max(-50.0, min(50.0, -self.steepness * (progress - center)))
        sigmoid = 1.0 / (1.0 + math.exp(exponent))
        return float(min_w + (max_w - min_w) * sigmoid)

    def get_weights(self, step, keys=None):
        if not self.is_active(step):
            return None

        progress = float(step) / float(self.total_warmup_epochs)
        weights = {}

        for key, center in self.stage_centers.items():
            min_w = self.min_weights.get(key, 1e-3)
            max_w = self.max_weights.get(key, 1.0)
            weights[key] = self._smooth_gate(progress, center, min_w, max_w)

        if self.couple_bc_with_az and "loss_bc" not in self.stage_centers:
            az_center = self.stage_centers.get("loss_a", 0.25)
            weights["loss_bc"] = self._smooth_gate(
                progress,
                az_center,
                self.min_weights.get("loss_bc", 0.05),
                self.max_weights.get("loss_bc", 1.0)
            )

        if keys is not None:
            return {k: float(weights.get(k, 1.0)) for k in keys}
        return weights

    def state_dict(self):
        return {
            "total_warmup_epochs": int(self.total_warmup_epochs),
            "stage_order": list(self.stage_order),
            "steepness": float(self.steepness),
            "min_weights": dict(self.min_weights),
            "max_weights": dict(self.max_weights),
            "couple_bc_with_az": bool(self.couple_bc_with_az),
            "stage_centers": dict(self.stage_centers)
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.total_warmup_epochs = int(state.get("total_warmup_epochs", self.total_warmup_epochs))
        if "stage_order" in state:
            self.stage_order = tuple(state["stage_order"])
        self.steepness = float(state.get("steepness", self.steepness))
        if "min_weights" in state:
            self.min_weights = {k: float(v) for k, v in state["min_weights"].items()}
        if "max_weights" in state:
            self.max_weights = {k: float(v) for k, v in state["max_weights"].items()}
        self.couple_bc_with_az = bool(state.get("couple_bc_with_az", self.couple_bc_with_az))
        if "stage_centers" in state:
            self.stage_centers = {k: float(v) for k, v in state["stage_centers"].items()}
