import math
import random


class ReLoBRaLoBalancer:
    def __init__(
        self,
        keys=("loss_pde", "loss_a", "loss_bn", "loss_ht"),
        alpha=0.999,
        temperature=0.1,
        rho=0.999,
        eps=1e-12
    ):
        self.keys = tuple(keys)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.rho = float(rho)
        self.eps = float(eps)
        self.lambdas = {k: 1.0 for k in self.keys}
        self.loss_init = None
        self.loss_prev = None
        self.step_count = 0

    def _to_scalar(self, val):
        if hasattr(val, "detach"):
            return float(val.detach().item())
        return float(val)

    def _scaled_softmax(self, ratios):
        k_count = len(ratios)
        if k_count == 0:
            return []
        if k_count == 1:
            return [1.0]
        tau = max(self.temperature, self.eps)
        scaled = [max(-50.0, min(50.0, r / tau)) for r in ratios]
        max_val = max(scaled)
        exps = [math.exp(s - max_val) for s in scaled]
        sum_exp = max(sum(exps), self.eps)
        return [float(k_count) * (e / sum_exp) for e in exps]

    def reset_anchors(self, reset_weights=False):
        self.loss_init = None
        self.loss_prev = None
        if reset_weights:
            self.lambdas = {k: 1.0 for k in self.keys}

    def get_weights(self, keys=None):
        target_keys = tuple(keys) if keys is not None else self.keys
        return {k: float(self.lambdas.get(k, 1.0)) for k in target_keys}

    def update(self, losses_dict, is_resampling_step=False):
        current_keys = tuple(losses_dict.keys())
        for k in current_keys:
            if k not in self.lambdas:
                self.lambdas[k] = 1.0

        current_vals = {k: self._to_scalar(losses_dict[k]) for k in current_keys}

        for v in current_vals.values():
            if not math.isfinite(v):
                return self.get_weights(current_keys)

        if self.loss_init is None or self.loss_prev is None:
            self.loss_init = dict(current_vals)
            self.loss_prev = dict(current_vals)
            self.step_count = 0
            return self.get_weights(current_keys)

        for k, v in current_vals.items():
            if k not in self.loss_init or self.loss_init[k] <= self.eps:
                self.loss_init[k] = v
            if k not in self.loss_prev or self.loss_prev[k] <= self.eps:
                self.loss_prev[k] = v

        if is_resampling_step:
            self.loss_prev = dict(current_vals)
            return self.get_weights(current_keys)

        active_keys = [k for k in current_keys if current_vals[k] > self.eps]
        if len(active_keys) <= 1:
            for k in current_keys:
                self.lambdas[k] = 1.0
            self.loss_prev = dict(current_vals)
            self.step_count += 1
            return self.get_weights(current_keys)

        short_ratios = [
            current_vals[k] / max(self.loss_prev[k], self.eps)
            for k in active_keys
        ]
        long_ratios = [
            current_vals[k] / max(self.loss_init[k], self.eps)
            for k in active_keys
        ]

        hat_short = self._scaled_softmax(short_ratios)
        hat_long = self._scaled_softmax(long_ratios)

        rho_sample = 1.0 if random.random() < self.rho else 0.0

        for idx, k in enumerate(active_keys):
            prev_lambda = float(self.lambdas.get(k, 1.0))
            hist_lambda = rho_sample * prev_lambda + (1.0 - rho_sample) * hat_long[idx]
            new_lambda = self.alpha * hist_lambda + (1.0 - self.alpha) * hat_short[idx]
            self.lambdas[k] = float(new_lambda)

        for k in current_keys:
            if k not in active_keys:
                self.lambdas[k] = 1.0

        self.loss_prev = dict(current_vals)
        self.step_count += 1
        return self.get_weights(current_keys)

    def state_dict(self):
        return {
            "keys": list(self.keys),
            "alpha": self.alpha,
            "temperature": self.temperature,
            "rho": self.rho,
            "eps": self.eps,
            "lambdas": dict(self.lambdas),
            "loss_init": dict(self.loss_init) if self.loss_init is not None else None,
            "loss_prev": dict(self.loss_prev) if self.loss_prev is not None else None,
            "step_count": int(self.step_count)
        }

    def load_state_dict(self, state):
        if not state:
            return
        if "keys" in state:
            self.keys = tuple(state["keys"])
        self.alpha = float(state.get("alpha", self.alpha))
        self.temperature = float(state.get("temperature", self.temperature))
        self.rho = float(state.get("rho", self.rho))
        self.eps = float(state.get("eps", self.eps))
        if "lambdas" in state and state["lambdas"] is not None:
            self.lambdas = {k: float(v) for k, v in state["lambdas"].items()}
        self.loss_init = (
            {k: float(v) for k, v in state["loss_init"].items()}
            if state.get("loss_init") is not None else None
        )
        self.loss_prev = (
            {k: float(v) for k, v in state["loss_prev"].items()}
            if state.get("loss_prev") is not None else None
        )
        self.step_count = int(state.get("step_count", 0))
