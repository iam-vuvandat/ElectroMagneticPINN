import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key


class FPIConvergenceTracker:
    def __init__(self, electro_magnetic_pinn, points_per_domain=2000, eps=1e-12):
        self.pinn = electro_magnetic_pinn
        self.points_per_domain = int(points_per_domain)
        self.eps = float(eps)
        self.eval_points = {}
        self.domain_names = {}
        self.prev_nu_star = {}
        self.prev_mu_r = {}
        self.history = {}

    def initialize(self):
        sim_domain = self.pinn.simulation_domain_engine_instance
        model = self.pinn.pinn_architecture_instance
        sampler = self.pinn.collocation_sampler_instance
        device = next(model.parameters()).device
        bb = (sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple)

        self.eval_points.clear()
        self.domain_names.clear()
        self.prev_nu_star.clear()
        self.prev_mu_r.clear()
        self.history.clear()

        for domain in sim_domain.domains_list:
            if getattr(domain, "domain_name", "") == "background":
                continue
            if not getattr(domain.physics_config, "is_nonlinear", False):
                continue

            net_key = resolve_domain_net_key(domain, model)
            if hasattr(model, "nu_networks") and net_key in model.nu_networks:
                pts = sampler.sample_domain_interior_points(
                    domain, bb, self.points_per_domain
                ).to(device).detach()
                self.eval_points[net_key] = pts
                self.domain_names[net_key] = getattr(domain, "domain_name", net_key)
                self.history[net_key] = {
                    "fpi_rel_l2_mu": [],
                    "fpi_rel_l2_nu": [],
                    "net_rel_l2_nu": [],
                    "net_rel_l2_mu": []
                }

    @staticmethod
    def relative_l2(current_tensor, reference_tensor, eps=1e-12):
        curr = current_tensor.reshape(-1)
        ref = reference_tensor.reshape(-1)
        num = (curr - ref).square().mean().sqrt()
        den = ref.square().mean().sqrt()
        return (num / (den + eps)).item()

    @torch.no_grad()
    def snapshot_before_fpi_step(self):
        if not self.eval_points:
            self.initialize()

        model = self.pinn.pinn_architecture_instance
        model.eval()
        for net_key, pts in self.eval_points.items():
            nu_star = model.forward_nu_network(net_key, pts)
            mu_r = 1.0 / torch.clamp(nu_star, min=1e-6)
            self.prev_nu_star[net_key] = nu_star.detach().clone()
            self.prev_mu_r[net_key] = mu_r.detach().clone()
        model.train()

    @torch.no_grad()
    def evaluate_after_fpi_step(self, fpi_step, net_fit_metrics=None):
        if not self.eval_points:
            return 0.0, 0.0, {}

        model = self.pinn.pinn_architecture_instance
        model.eval()

        per_domain_metrics = {}
        global_max_fpi_mu = 0.0
        global_max_fpi_nu = 0.0

        for net_key, pts in self.eval_points.items():
            nu_curr = model.forward_nu_network(net_key, pts)
            mu_curr = 1.0 / torch.clamp(nu_curr, min=1e-6)

            nu_prev = self.prev_nu_star[net_key]
            mu_prev = self.prev_mu_r[net_key]

            err_fpi_mu = self.relative_l2(mu_curr, mu_prev, self.eps)
            err_fpi_nu = self.relative_l2(nu_curr, nu_prev, self.eps)

            fit_info = net_fit_metrics.get(net_key, {}) if net_fit_metrics else {}
            err_net_nu = float(fit_info.get("net_rel_l2_nu", 0.0))
            err_net_mu = float(fit_info.get("net_rel_l2_mu", 0.0))

            self.history[net_key]["fpi_rel_l2_mu"].append(err_fpi_mu)
            self.history[net_key]["fpi_rel_l2_nu"].append(err_fpi_nu)
            self.history[net_key]["net_rel_l2_nu"].append(err_net_nu)
            self.history[net_key]["net_rel_l2_mu"].append(err_net_mu)

            d_name = self.domain_names[net_key]
            per_domain_metrics[d_name] = {
                "fpi_rel_l2_mu": err_fpi_mu,
                "fpi_rel_l2_nu": err_fpi_nu,
                "net_rel_l2_nu": err_net_nu,
                "net_rel_l2_mu": err_net_mu
            }

            global_max_fpi_mu = max(global_max_fpi_mu, err_fpi_mu)
            global_max_fpi_nu = max(global_max_fpi_nu, err_fpi_nu)

            print(
                f"[FPI Step {fpi_step:02d} | {d_name}] "
                f"FPI_RelL2(mu_r) = {err_fpi_mu:.4e} ({err_fpi_mu * 100.0:.2f}%) | "
                f"FPI_RelL2(nu*) = {err_fpi_nu:.4e} | "
                f"Net_RelL2(mu_r/nu*) = {err_net_mu:.4e}/{err_net_nu:.4e}"
            )

        print(
            f"[FPI Step {fpi_step:02d} | GLOBAL MAX] "
            f"Max_FPI_RelL2(mu_r) = {global_max_fpi_mu:.4e} ({global_max_fpi_mu * 100.0:.2f}%) | "
            f"Max_FPI_RelL2(nu*) = {global_max_fpi_nu:.4e}"
        )

        model.train()
        return global_max_fpi_mu, global_max_fpi_nu, per_domain_metrics

    def evaluate_against_bh_reference(self, points_per_domain=None):
        sim_domain = self.pinn.simulation_domain_engine_instance
        model = self.pinn.pinn_architecture_instance
        sampler = self.pinn.collocation_sampler_instance
        device = next(model.parameters()).device
        bb = (sim_domain.x_boundaries_tuple, sim_domain.y_boundaries_tuple)
        n_pts = int(points_per_domain) if points_per_domain is not None else self.points_per_domain

        model.eval()
        results = {}
        max_rel_l2_mu = 0.0
        max_rel_l2_nu = 0.0

        for domain in sim_domain.domains_list:
            if getattr(domain, "domain_name", "") == "background":
                continue
            if not getattr(domain.physics_config, "is_nonlinear", False):
                continue

            net_key = resolve_domain_net_key(domain, model)
            if not (hasattr(model, "nu_networks") and net_key in model.nu_networks):
                continue

            if net_key in self.eval_points and points_per_domain is None:
                pts = self.eval_points[net_key]
            else:
                pts = sampler.sample_domain_interior_points(domain, bb, n_pts).to(device).detach()

            pts_grad = pts.detach().clone().requires_grad_(True)
            Az_star = model.forward_network(net_key, pts_grad)
            grad_Az = torch.autograd.grad(
                outputs=Az_star,
                inputs=pts_grad,
                grad_outputs=torch.ones_like(Az_star),
                create_graph=False
            )[0]
            Bx_star = grad_Az[:, 1:2] * sim_domain.L0
            By_star = -grad_Az[:, 0:1] * sim_domain.L0
            B_mag_phys = torch.sqrt(Bx_star**2 + By_star**2 + 1e-12) * sim_domain.B0

            with torch.no_grad():
                nu_ref_phys = domain.evaluate_reluctivity(
                    pts_grad, B_magnitude_tensor=B_mag_phys, is_nonlinear=True
                )
                nu_ref_star = torch.clamp(nu_ref_phys / sim_domain.nu0, min=1e-6)
                mu_ref = 1.0 / nu_ref_star

                nu_pred_star = torch.clamp(model.forward_nu_network(net_key, pts), min=1e-6)
                mu_pred = 1.0 / nu_pred_star

                rel_l2_nu = self.relative_l2(nu_pred_star, nu_ref_star, self.eps)
                rel_l2_mu = self.relative_l2(mu_pred, mu_ref, self.eps)

            d_name = getattr(domain, "domain_name", net_key)
            results[d_name] = {
                "rel_l2_mu_r": rel_l2_mu,
                "rel_l2_nu_star": rel_l2_nu
            }
            max_rel_l2_mu = max(max_rel_l2_mu, rel_l2_mu)
            max_rel_l2_nu = max(max_rel_l2_nu, rel_l2_nu)

            print(
                f"[Permeability L2 | {d_name}] "
                f"RelL2(mu_r) = {rel_l2_mu:.4e} ({rel_l2_mu * 100.0:.2f}%) | "
                f"RelL2(nu*) = {rel_l2_nu:.4e} ({rel_l2_nu * 100.0:.2f}%)"
            )

        summary = {
            "per_domain": results,
            "max_rel_l2_mu_r": max_rel_l2_mu,
            "max_rel_l2_nu_star": max_rel_l2_nu
        }
        print(
            f"[Permeability L2 | GLOBAL MAX] "
            f"Max_RelL2(mu_r) = {max_rel_l2_mu:.4e} ({max_rel_l2_mu * 100.0:.2f}%) | "
            f"Max_RelL2(nu*) = {max_rel_l2_nu:.4e} ({max_rel_l2_nu * 100.0:.2f}%)"
        )
        model.train()
        return summary

    def state_dict(self):
        return {
            "points_per_domain": int(self.points_per_domain),
            "eps": float(self.eps),
            "eval_points": {k: v.detach().cpu() for k, v in self.eval_points.items()},
            "domain_names": dict(self.domain_names),
            "prev_nu_star": {k: v.detach().cpu() for k, v in self.prev_nu_star.items()},
            "prev_mu_r": {k: v.detach().cpu() for k, v in self.prev_mu_r.items()},
            "history": {
                k: {mk: list(mv) for mk, mv in v.items()}
                for k, v in self.history.items()
            }
        }

    def load_state_dict(self, state):
        if not state:
            return
        model = self.pinn.pinn_architecture_instance
        device = next(model.parameters()).device if model is not None else torch.device("cpu")
        self.points_per_domain = int(state.get("points_per_domain", self.points_per_domain))
        self.eps = float(state.get("eps", self.eps))
        self.eval_points = {
            k: v.to(device) for k, v in state.get("eval_points", {}).items()
        }
        self.domain_names = dict(state.get("domain_names", {}))
        self.prev_nu_star = {
            k: v.to(device) for k, v in state.get("prev_nu_star", {}).items()
        }
        self.prev_mu_r = {
            k: v.to(device) for k, v in state.get("prev_mu_r", {}).items()
        }
        self.history = {
            k: {mk: list(mv) for mk, mv in v.items()}
            for k, v in state.get("history", {}).items()
        }