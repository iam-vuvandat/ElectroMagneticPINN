import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    print(f"[DEBUG] Device: GPU ({torch.cuda.get_device_name(0)})")
else:
    print("[DEBUG] Device: CPU")

MU0, M0, L0 = 4.0 * math.pi * 1.0e-7, 1.0e6, 1.0
A0, B0 = MU0 * M0 * L0, MU0 * M0

IRON_MATERIAL_NAME = "steel_1008"
BH_EXTRAPOLATION, BH_B_EPS = "linear", 1.0e-6

DOMAIN_EXTENT = 2.0
LENGTH_X, WIDTH_Y, STEEPNESS = 0.30, 0.12, 45.0
MAGNET_OFFSET_X = 1.15 * LENGTH_X
SPAN_X = MAGNET_OFFSET_X + LENGTH_X
YOKE_LENGTH_X, YOKE_WIDTH_Y, YOKE_OFFSET_Y = 1.10 * SPAN_X, WIDTH_Y, -(2.0 * WIDTH_Y)
LEFT_POLARITY, RIGHT_POLARITY = -1.0, +1.0

class Iron:
    _DATABASE = {
        "M350-50A": {
            "B_data": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.956, 2.1, 2.2, 2.2701, 2.4], dtype=np.float64),
            "H_data": np.array([0.0, 34.8, 46.0, 53.7, 60.6, 67.4, 74.6, 82.6, 91.8, 103.0, 119.0, 141.0, 178.0, 250.0, 455.0, 1180.0, 3020.0, 6100.0, 10700.0, 25000.0, 35000.0, 75000.0, 115000.0, 150000.0, 229580.0], dtype=np.float64)
        },
        "steel_1008": {
            "B_data": np.array([0.0, 0.2402, 0.8654, 1.1106, 1.2458, 1.331, 1.5, 1.6, 1.683, 1.741, 1.78, 1.905, 2.025, 2.085, 2.13, 2.165, 2.28, 2.485, 2.5851], dtype=np.float64),
            "H_data": np.array([0.0, 159.2, 318.3, 477.5, 636.6, 795.8, 1591.5, 3183.1, 4774.6, 6366.2, 7957.7, 15915.5, 31831.0, 47746.5, 63662.0, 79577.5, 159155.0, 318310.0, 397887.0], dtype=np.float64)
        }
    }

    def __init__(self, name: str):
        if name not in self._DATABASE: raise ValueError(f"Iron '{name}' not found.")
        data = self._DATABASE[name]
        self.name = name
        self.B_H_curve = {"B_data": data["B_data"].copy(), "H_data": data["H_data"].copy()}
        self._validate()

    def _validate(self):
        B, H = self.B_H_curve["B_data"], self.B_H_curve["H_data"]
        if len(B) != len(H) or len(B) < 3: raise ValueError("Invalid B-H data.")
        if not np.isclose(B[0], 0.0) or not np.isclose(H[0], 0.0): raise ValueError("Curve must start at 0.")
        if np.any(np.diff(B) <= 0.0) or np.any(np.diff(H) < 0.0): raise ValueError("Invalid B-H monotonicity.")

def compute_pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(x)
    if n == 2:
        d[:] = delta[0]
        return d
    for i in range(1, n - 1):
        d_left, d_right = delta[i - 1], delta[i]
        if d_left == 0.0 or d_right == 0.0 or d_left * d_right <= 0.0:
            d[i] = 0.0
            continue
        h_left, h_right = h[i - 1], h[i]
        w1, w2 = 2.0 * h_right + h_left, h_right + 2.0 * h_left
        d[i] = (w1 + w2) / (w1 / d_left + w2 / d_right)

    def endpoint_slope(h0, h1, delta0, delta1):
        slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
        if np.sign(slope) != np.sign(delta0): slope = 0.0
        elif np.sign(delta0) != np.sign(delta1) and abs(slope) > 3.0 * abs(delta0): slope = 3.0 * delta0
        return slope

    d[0], d[-1] = endpoint_slope(h[0], h[1], delta[0], delta[1]), endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d

class BHCurve:
    def __init__(self, material_name=IRON_MATERIAL_NAME, device=DEVICE, dtype=DTYPE):
        self.material = Iron(material_name)
        B_np, H_np = self.material.B_H_curve["B_data"], self.material.B_H_curve["H_data"]
        dH_dB_np = compute_pchip_slopes(B_np, H_np)
        self.B_data = torch.tensor(B_np, device=device, dtype=dtype)
        self.H_data = torch.tensor(H_np, device=device, dtype=dtype)
        self.dH_dB_data = torch.tensor(dH_dB_np, device=device, dtype=dtype)
        self.B_min, self.B_max = float(B_np[0]), float(B_np[-1])
        self.H_min, self.H_max = float(H_np[0]), float(H_np[-1])
        self.initial_slope, self.final_slope = float(dH_dB_np[0]), float(dH_dB_np[-1])
        self.mu_r_initial = 1.0 / (MU0 * self.initial_slope)

    def H_from_B(self, B):
        original_shape = B.shape
        B_flat = B.reshape(-1)
        B_eval = torch.clamp(B_flat, min=self.B_min, max=self.B_max)
        idx = torch.clamp(torch.searchsorted(self.B_data, B_eval, right=False), 1, len(self.B_data) - 1)
        B0_val, B1_val = self.B_data[idx - 1], self.B_data[idx]
        H0_val, H1_val = self.H_data[idx - 1], self.H_data[idx]
        d0_val, d1_val = self.dH_dB_data[idx - 1], self.dH_dB_data[idx]
        h = B1_val - B0_val
        t = (B_eval - B0_val) / (h + 1.0e-12)
        t2, t3 = t * t, t * t * t
        H_inside = (2.0 * t3 - 3.0 * t2 + 1.0) * H0_val + (t3 - 2.0 * t2 + t) * h * d0_val + (-2.0 * t3 + 3.0 * t2) * H1_val + (t3 - t2) * h * d1_val
        H_below = self.H_data[0] + self.dH_dB_data[0] * (B_flat - self.B_data[0])
        H_above = self.H_data[-1] + self.dH_dB_data[-1] * (B_flat - self.B_data[-1])
        H = torch.where(B_flat < self.B_min, H_below, torch.where(B_flat > self.B_max, H_above, H_inside))
        return H.reshape(original_shape)

    def relative_reluctivity(self, B):
        B_safe = torch.clamp(B, min=1.0e-12)
        nu_r = MU0 * self.H_from_B(B_safe) / B_safe
        return torch.where(B < BH_B_EPS, torch.full_like(B, MU0 * self.initial_slope), nu_r)

    def relative_permeability(self, B):
        return 1.0 / torch.clamp(self.relative_reluctivity(B), min=1.0e-12)

class MagnetizationSource:
    def __init__(self, length_x=LENGTH_X, width_y=WIDTH_Y, offset_x=MAGNET_OFFSET_X, steepness=STEEPNESS, left_polarity=LEFT_POLARITY, right_polarity=RIGHT_POLARITY):
        self.lx, self.wy, self.offset_x, self.k = float(length_x), float(width_y), float(offset_x), float(steepness)
        self.left_polarity, self.right_polarity = float(left_polarity), float(right_polarity)

    def _interval(self, z, z_min, z_max):
        return torch.sigmoid(self.k * (z - z_min)) - torch.sigmoid(self.k * (z - z_max))

    def _interval_derivative(self, z, z_min, z_max):
        s_min, s_max = torch.sigmoid(self.k * (z - z_min)), torch.sigmoid(self.k * (z - z_max))
        return self.k * s_min * (1.0 - s_min) - self.k * s_max * (1.0 - s_max)

    def _components(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return self._interval(x, -self.offset_x - self.lx, -self.offset_x + self.lx), self._interval(x, self.offset_x - self.lx, self.offset_x + self.lx), self._interval(y, -self.wy, self.wy)

    def dimensionless(self, xy):
        left_x, right_x, y_mask = self._components(xy)
        return (self.left_polarity * left_x + self.right_polarity * right_x) * y_mask

    def magnet_mask(self, xy):
        left_x, right_x, y_mask = self._components(xy)
        return (left_x + right_x) * y_mask

    def physical(self, xy):
        return M0 * self.dimensionless(xy)

    def dmx_dx_dimensionless(self, xy):
        left_x, right_x, y_mask = self._components(xy)
        dleft_dx = self._interval_derivative(xy[:, 0:1], -self.offset_x - self.lx, -self.offset_x + self.lx)
        dright_dx = self._interval_derivative(xy[:, 0:1], self.offset_x - self.lx, self.offset_x + self.lx)
        return (self.left_polarity * dleft_dx + self.right_polarity * dright_dx) * y_mask

    def dmx_dy_dimensionless(self, xy):
        left_x, right_x, _ = self._components(xy)
        return (self.left_polarity * left_x + self.right_polarity * right_x) * self._interval_derivative(xy[:, 1:2], -self.wy, self.wy)

    def dmx_dx_physical(self, xy): return (M0 / L0) * self.dmx_dx_dimensionless(xy)
    def dmx_dy_physical(self, xy): return (M0 / L0) * self.dmx_dy_dimensionless(xy)

class MaterialPermeability:
    def __init__(self, yoke_lx=YOKE_LENGTH_X, yoke_wy=YOKE_WIDTH_Y, yoke_offset_y=YOKE_OFFSET_Y, steepness=STEEPNESS, iron_name=IRON_MATERIAL_NAME):
        self.lx, self.wy, self.y0, self.k, self.nu_air = float(yoke_lx), float(yoke_wy), float(yoke_offset_y), float(steepness), 1.0
        self.bh_curve = BHCurve(iron_name, device=DEVICE, dtype=DTYPE)
        self.mur_linear, self.nu_iron_linear, self.nonlinear_alpha = self.bh_curve.mu_r_initial, 1.0 / self.bh_curve.mu_r_initial, 0.0

    @property
    def iron_data(self): return self.bh_curve.material

    @property
    def B_max_data(self): return self.bh_curve.B_max

    def set_nonlinear_alpha(self, alpha): 
        self.nonlinear_alpha = float(np.clip(alpha, 0.0, 1.0))

    def iron_mask(self, xy):
        sig_x = torch.sigmoid(self.k * (xy[:, 0:1] + self.lx)) - torch.sigmoid(self.k * (xy[:, 0:1] - self.lx))
        sig_y = torch.sigmoid(self.k * (xy[:, 1:2] - self.y0 + self.wy)) - torch.sigmoid(self.k * (xy[:, 1:2] - self.y0 - self.wy))
        return sig_x * sig_y

    def B_magnitude(self, dAz_dx, dAz_dy): 
        return B0 * torch.sqrt(dAz_dx**2 + dAz_dy**2 + 1.0e-12)

    def nonlinear_reluctivity(self, B_mag): 
        return self.bh_curve.relative_reluctivity(B_mag)

    def nu_r_dimensionless(self, xy, dAz_dx=None, dAz_dy=None):
        mask_iron = self.iron_mask(xy)
        if self.nonlinear_alpha <= 0.0 or dAz_dx is None or dAz_dy is None:
            nu_iron = torch.full_like(mask_iron, self.nu_iron_linear)
        else:
            nu_iron = (1.0 - self.nonlinear_alpha) * self.nu_iron_linear + self.nonlinear_alpha * self.nonlinear_reluctivity(self.B_magnitude(dAz_dx, dAz_dy))
        return self.nu_air + (nu_iron - self.nu_air) * mask_iron

    def mu_r_dimensionless(self, xy, dAz_dx=None, dAz_dy=None):
        return 1.0 / torch.clamp(self.nu_r_dimensionless(xy, dAz_dx, dAz_dy), min=1.0e-12)

    def H_magnitude(self, B_mag, nu_r): 
        return B_mag * nu_r / MU0

class MagneticPINN(nn.Module):
    def __init__(self, hidden_layers=6, neurons=128, hard_boundary=True):
        super().__init__()
        self.hard_boundary = hard_boundary
        layers = [nn.Linear(2, neurons), nn.SiLU()]
        for _ in range(hidden_layers - 1): 
            layers += [nn.Linear(neurons, neurons), nn.SiLU()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def boundary_factor(self, xy):
        return (1.0 - (xy[:, 0:1] / DOMAIN_EXTENT)**2) * (1.0 - (xy[:, 1:2] / DOMAIN_EXTENT)**2)

    def forward(self, xy):
        raw = self.net(xy / DOMAIN_EXTENT)
        return self.boundary_factor(xy) * raw if self.hard_boundary else raw

def create_sobol_engine(seed=SEED):
    return torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)

def sample_collocation_points(num_points, device=DEVICE, engine=None):
    engine = engine if engine is not None else create_sobol_engine()
    uv = engine.draw(num_points).to(device=device, dtype=DTYPE)
    
    num_core = int(0.75 * num_points)
    x_core = -0.8 + 1.6 * uv[:num_core, 0:1]
    y_core = -0.6 + 1.0 * uv[:num_core, 1:2]
    
    x_full = -DOMAIN_EXTENT + 2.0 * DOMAIN_EXTENT * uv[num_core:, 0:1]
    y_full = -DOMAIN_EXTENT + 2.0 * DOMAIN_EXTENT * uv[num_core:, 1:2]
    
    xy = torch.cat([torch.cat([x_core, y_core], dim=1), torch.cat([x_full, y_full], dim=1)], dim=0)
    xy.requires_grad_(True)
    return xy

def sample_boundary_points(num_points, device=DEVICE):
    t = create_sobol_engine(seed=SEED + 20).draw(4 * num_points).to(device=device, dtype=DTYPE)
    t1, t2, t3, t4 = t[:num_points], t[num_points:2 * num_points], t[2 * num_points:3 * num_points], t[3 * num_points:4 * num_points]
    x_bottom, x_top = -DOMAIN_EXTENT + 2.0 * DOMAIN_EXTENT * t1[:, 0:1], -DOMAIN_EXTENT + 2.0 * DOMAIN_EXTENT * t2[:, 0:1]
    y_left, y_right = -DOMAIN_EXTENT + 2.0 * DOMAIN_EXTENT * t3[:, 0:1], -DOMAIN_EXTENT + 2.0 * DOMAIN_EXTENT * t4[:, 0:1]
    return torch.cat([
        torch.cat([x_bottom, torch.full_like(x_bottom, -DOMAIN_EXTENT)], dim=1),
        torch.cat([x_top, torch.full_like(x_top, DOMAIN_EXTENT)], dim=1),
        torch.cat([torch.full_like(y_left, -DOMAIN_EXTENT), y_left], dim=1),
        torch.cat([torch.full_like(y_right, DOMAIN_EXTENT), y_right], dim=1)
    ], dim=0)

def compute_pde_loss(model, xy_collocation, source, material):
    Az_star = model(xy_collocation)
    grads = torch.autograd.grad(outputs=Az_star, inputs=xy_collocation, grad_outputs=torch.ones_like(Az_star), create_graph=True)[0]
    dAz_dx, dAz_dy = grads[:, 0:1], grads[:, 1:2]
    nu_r = material.nu_r_dimensionless(xy_collocation, dAz_dx, dAz_dy)
    flux_x, flux_y = nu_r * dAz_dx, nu_r * dAz_dy
    div_flux_x = torch.autograd.grad(outputs=flux_x, inputs=xy_collocation, grad_outputs=torch.ones_like(flux_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    div_flux_y = torch.autograd.grad(outputs=flux_y, inputs=xy_collocation, grad_outputs=torch.ones_like(flux_y), create_graph=True)[0][:, 1:2]
    residual = div_flux_x + div_flux_y - source.dmx_dy_dimensionless(xy_collocation)
    return torch.mean(residual**2), residual

def evaluate_fields(model, source, material, resolution=160, device=DEVICE):
    was_training = model.training
    model.eval()
    X_star, Y_star = np.meshgrid(np.linspace(-DOMAIN_EXTENT, DOMAIN_EXTENT, resolution, dtype=np.float32), np.linspace(-DOMAIN_EXTENT, DOMAIN_EXTENT, resolution, dtype=np.float32))
    xy = torch.tensor(np.column_stack((X_star.ravel(), Y_star.ravel())).astype(np.float32), device=device, dtype=DTYPE, requires_grad=True)
    Az_star = model(xy)
    grads = torch.autograd.grad(outputs=Az_star, inputs=xy, grad_outputs=torch.ones_like(Az_star), create_graph=False)[0]
    dAz_dx, dAz_dy = grads[:, 0:1], grads[:, 1:2]
    Az, Bx, By = (A0 * Az_star).detach().cpu().numpy().reshape(X_star.shape), (B0 * dAz_dy).detach().cpu().numpy().reshape(X_star.shape), (-B0 * dAz_dx).detach().cpu().numpy().reshape(X_star.shape)
    with torch.no_grad():
        xy_detach, dx_detach, dy_detach = xy.detach(), dAz_dx.detach(), dAz_dy.detach()
        nu_r_tensor = material.nu_r_dimensionless(xy_detach, dx_detach, dy_detach)
        Mx = source.physical(xy_detach).cpu().numpy().reshape(X_star.shape)
        dMx_dy = source.dmx_dy_physical(xy_detach).cpu().numpy().reshape(X_star.shape)
        rho_m = -source.dmx_dx_physical(xy_detach).cpu().numpy().reshape(X_star.shape)
        magnet_mask = source.magnet_mask(xy_detach).cpu().numpy().reshape(X_star.shape)
        iron_mask = material.iron_mask(xy_detach).cpu().numpy().reshape(X_star.shape)
        Mur = (1.0 / torch.clamp(nu_r_tensor, min=1.0e-12)).cpu().numpy().reshape(X_star.shape)
        Hmag = material.H_magnitude(material.B_magnitude(dx_detach, dy_detach), nu_r_tensor).cpu().numpy().reshape(X_star.shape)
    if was_training: model.train()
    return {"X": L0 * X_star, "Y": L0 * Y_star, "Az": Az, "Bx": Bx, "By": By, "Bmag": np.sqrt(Bx**2 + By**2), "Mx": Mx, "dMx_dy": dMx_dy, "rho_m": rho_m, "Mur": Mur, "Hmag": Hmag, "magnet_mask": magnet_mask, "iron_mask": iron_mask}

def evaluate_BH_state(model, material, num_points=5000):
    xy = sample_collocation_points(num_points, device=DEVICE, engine=create_sobol_engine(seed=SEED + 100))
    grads = torch.autograd.grad(outputs=model(xy), inputs=xy, grad_outputs=torch.ones_like(model(xy)), create_graph=False)[0]
    Bmag = material.B_magnitude(grads[:, 0:1], grads[:, 1:2])
    return Bmag.detach(), material.H_magnitude(Bmag, material.nu_r_dimensionless(xy, grads[:, 0:1], grads[:, 1:2])).detach()

def compute_boundary_error(model, num_points=1000):
    with torch.no_grad(): return torch.max(torch.abs(model(sample_boundary_points(num_points, device=DEVICE)))).item()

def compute_pde_validation(model, source, material, num_points=5000):
    loss, residual = compute_pde_loss(model, sample_collocation_points(num_points, device=DEVICE, engine=create_sobol_engine(seed=SEED + 200)), source, material)
    Bmag, Hmag = evaluate_BH_state(model, material, num_points=num_points)
    return {"pde_mse": loss.item(), "pde_rms": torch.sqrt(torch.mean(residual**2)).item(), "pde_max": torch.max(torch.abs(residual)).item(), "Bmax": torch.max(Bmag).item(), "Hmax": torch.max(Hmag).item(), "bh_outside_fraction": torch.mean((Bmag > material.B_max_data).to(DTYPE)).item()}

def compute_boundary_flux(model, resolution=400, device=DEVICE):
    model.eval()
    t = torch.linspace(-DOMAIN_EXTENT, DOMAIN_EXTENT, resolution, device=device, dtype=DTYPE).reshape(-1, 1)
    def B_at(points):
        points = points.clone().requires_grad_(True)
        grads = torch.autograd.grad(outputs=model(points), inputs=points, grad_outputs=torch.ones_like(model(points)), create_graph=False)[0]
        return (B0 * grads[:, 1:2]).detach().cpu().numpy().ravel(), (-B0 * grads[:, 0:1]).detach().cpu().numpy().ravel()
    bx_bottom, by_bottom = B_at(torch.cat([t, torch.full_like(t, -DOMAIN_EXTENT)], dim=1))
    bx_top, by_top = B_at(torch.cat([t, torch.full_like(t, DOMAIN_EXTENT)], dim=1))
    bx_left, by_left = B_at(torch.cat([torch.full_like(t, -DOMAIN_EXTENT), t], dim=1))
    bx_right, by_right = B_at(torch.cat([torch.full_like(t, DOMAIN_EXTENT), t], dim=1))
    t_np = t.detach().cpu().numpy().ravel()
    p_b, p_t, p_l, p_r = np.trapz(-by_bottom, t_np), np.trapz(by_top, t_np), np.trapz(-bx_left, t_np), np.trapz(bx_right, t_np)
    return {"bottom": p_b, "top": p_t, "left": p_l, "right": p_r, "total": p_b + p_t + p_l + p_r}

def report_diagnostics(model, source, material, num_validation=5000):
    model.eval()
    err, val, fields = compute_boundary_error(model, 1000), compute_pde_validation(model, source, material, num_validation), evaluate_fields(model, source, material, 100, DEVICE)
    flux = compute_boundary_flux(model, 300, DEVICE)
    print("\n========================================\n             PINN DIAGNOSTICS\n========================================")
    print(f"Device                 : {DEVICE}\nMaterial               : {material.iron_data.name}\nNonlinear alpha        : {material.nonlinear_alpha:.4f}\nInitial mu_r           : {material.mur_linear:.6f}\nBH B range             : 0 -> {material.B_max_data:.6f} T\nBoundary max |Az|      : {err:.8e} Wb/m\n----------------------------------------")
    print(f"PDE MSE                : {val['pde_mse']:.8e}\nPDE RMS residual       : {val['pde_rms']:.8e}\nPDE max |residual|     : {val['pde_max']:.8e}\n----------------------------------------")
    print(f"Bmax                   : {val['Bmax']:.8e} T\nHmax                   : {val['Hmax']:.8e} A/m\nmu_r min               : {float(np.min(fields['Mur'])):.8e}\nmu_r max               : {float(np.max(fields['Mur'])):.8e}\nB > BH max fraction    : {100.0 * val['bh_outside_fraction']:.4f}%\n----------------------------------------")
    print(f"Flux bottom            : {flux['bottom']:.8e} Wb/m\nFlux top               : {flux['top']:.8e} Wb/m\nFlux left              : {flux['left']:.8e} Wb/m\nFlux right             : {flux['right']:.8e} Wb/m\nClosed-boundary flux   : {flux['total']:.8e} Wb/m\n========================================\n")

def train_pinn_curriculum(epochs_linear=3000, lbfgs_linear=150, epochs_nonlinear=1800, lbfgs_nonlinear=100, num_collocation=10000, lbfgs_collocation=5000, hidden_layers=6, neurons=128, linear_lr=1.0e-3, nonlinear_lr=2.0e-4, nonlinear_ramp_fraction=0.40, print_every_linear=250, print_every_nonlinear=100, grad_clip_linear=1.0, grad_clip_nonlinear=0.5, adaptive_refinement=True, adaptive_interval=500, adaptive_ratio=0.05):
    model, source, material = MagneticPINN(hidden_layers, neurons, True).to(DEVICE), MagnetizationSource(), MaterialPermeability()
    history = {"adam_linear": [], "lbfgs_linear": [], "adam_nonlinear": [], "lbfgs_nonlinear": [], "alpha": [], "lr_nonlinear": []}
    
    num_random = int(num_collocation * (1.0 - adaptive_ratio)) if adaptive_refinement else num_collocation
    num_adapt = num_collocation - num_random
    xy_adapt = None
    engine_linear, engine_pool = create_sobol_engine(seed=SEED), create_sobol_engine(seed=SEED+999) if adaptive_refinement else None

    print("\n" + "=" * 56 + "\nPHASE 1: LINEAR\n" + "=" * 56)
    material.set_nonlinear_alpha(0.0)
    adam1 = optim.Adam(model.parameters(), lr=linear_lr)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(adam1, T_max=epochs_linear, eta_min=1.0e-6)
    
    model.train()
    for epoch in range(1, epochs_linear + 1):
        if adaptive_refinement and (epoch == 1 or epoch % adaptive_interval == 0):
            model.eval()
            xy_pool = sample_collocation_points(num_collocation * 5, device=DEVICE, engine=engine_pool)
            _, res = compute_pde_loss(model, xy_pool, source, material)
            res_mag = torch.abs(res).squeeze().detach()
            _, idx = torch.topk(res_mag, num_adapt)
            xy_adapt = xy_pool[idx].detach()
            del xy_pool, res, res_mag
            model.train()

        xy_rand = sample_collocation_points(num_random, device=DEVICE, engine=engine_linear)
        xy = torch.cat([xy_rand, xy_adapt], dim=0).requires_grad_(True) if adaptive_refinement and xy_adapt is not None else xy_rand

        adam1.zero_grad(set_to_none=True)
        loss, _ = compute_pde_loss(model, xy, source, material)
        loss.backward()
        if grad_clip_linear is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_linear)
        adam1.step(); scheduler1.step(); history["adam_linear"].append(loss.item())
        if epoch == 1 or epoch % print_every_linear == 0: print(f"[Linear Adam] Epoch {epoch:05d} | PDE = {loss.item():.6e} | LR = {adam1.param_groups[0]['lr']:.3e}")

    print("\n[Linear L-BFGS]")
    xy_fixed_linear = sample_collocation_points(lbfgs_collocation, device=DEVICE, engine=create_sobol_engine(seed=SEED + 1))
    if adaptive_refinement and xy_adapt is not None: xy_fixed_linear = torch.cat([xy_fixed_linear, xy_adapt.clone()], dim=0).requires_grad_(True)
    lbfgs1 = optim.LBFGS(model.parameters(), lr=0.8, max_iter=lbfgs_linear, max_eval=max(lbfgs_linear + 50, int(1.25 * lbfgs_linear)), history_size=50, tolerance_grad=1.0e-8, tolerance_change=1.0e-10, line_search_fn="strong_wolfe")
    lbfgs_step_1 = [0]
    def closure_linear():
        lbfgs1.zero_grad(set_to_none=True)
        loss, _ = compute_pde_loss(model, xy_fixed_linear, source, material)
        loss.backward(); lbfgs_step_1[0] += 1; history["lbfgs_linear"].append(loss.item())
        if lbfgs_step_1[0] == 1 or lbfgs_step_1[0] % 25 == 0: print(f"[Linear L-BFGS] Step {lbfgs_step_1[0]:05d} | PDE = {loss.item():.6e}")
        return loss
    lbfgs1.step(closure_linear)

    print("\n" + "=" * 56 + "\nPHASE 2: NONLINEAR CONTINUATION\n" + "=" * 56)
    adam2 = optim.Adam(model.parameters(), lr=nonlinear_lr)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(adam2, T_max=epochs_nonlinear, eta_min=1.0e-6)
    engine_nonlinear, ramp_epochs = create_sobol_engine(seed=SEED + 2), max(1, int(nonlinear_ramp_fraction * epochs_nonlinear))
    
    model.train()
    for epoch in range(1, epochs_nonlinear + 1):
        if adaptive_refinement and (epoch == 1 or epoch % adaptive_interval == 0):
            model.eval()
            xy_pool = sample_collocation_points(num_collocation * 5, device=DEVICE, engine=engine_pool)
            _, res = compute_pde_loss(model, xy_pool, source, material)
            res_mag = torch.abs(res).squeeze().detach()
            _, idx = torch.topk(res_mag, num_adapt)
            xy_adapt = xy_pool[idx].detach()
            del xy_pool, res, res_mag
            model.train()

        xy_rand = sample_collocation_points(num_random, device=DEVICE, engine=engine_nonlinear)
        xy = torch.cat([xy_rand, xy_adapt], dim=0).requires_grad_(True) if adaptive_refinement and xy_adapt is not None else xy_rand

        adam2.zero_grad(set_to_none=True)
        alpha = 1.0 if epoch > ramp_epochs else (1.0 if ramp_epochs == 1 else (epoch - 1) / (ramp_epochs - 1))
        material.set_nonlinear_alpha(alpha)
        loss, _ = compute_pde_loss(model, xy, source, material)
        loss.backward()
        if grad_clip_nonlinear is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_nonlinear)
        adam2.step(); scheduler2.step()
        history["adam_nonlinear"].append(loss.item()); history["alpha"].append(alpha); history["lr_nonlinear"].append(adam2.param_groups[0]["lr"])
        if epoch == 1 or epoch % print_every_nonlinear == 0: print(f"[Nonlinear Adam] Epoch {epoch:05d} | alpha = {alpha:.3f} | PDE = {loss.item():.6e} | LR = {adam2.param_groups[0]['lr']:.3e}")

    print("\n[Nonlinear L-BFGS]")
    material.set_nonlinear_alpha(1.0)
    xy_fixed_nonlinear = sample_collocation_points(lbfgs_collocation, device=DEVICE, engine=create_sobol_engine(seed=SEED + 3))
    if adaptive_refinement and xy_adapt is not None: xy_fixed_nonlinear = torch.cat([xy_fixed_nonlinear, xy_adapt.clone()], dim=0).requires_grad_(True)
    lbfgs2 = optim.LBFGS(model.parameters(), lr=0.5, max_iter=lbfgs_nonlinear, max_eval=max(lbfgs_nonlinear + 50, int(1.25 * lbfgs_nonlinear)), history_size=50, tolerance_grad=1.0e-8, tolerance_change=1.0e-10, line_search_fn="strong_wolfe")
    lbfgs_step_2 = [0]
    def closure_nonlinear():
        lbfgs2.zero_grad(set_to_none=True)
        loss, _ = compute_pde_loss(model, xy_fixed_nonlinear, source, material)
        loss.backward(); lbfgs_step_2[0] += 1; history["lbfgs_nonlinear"].append(loss.item())
        if lbfgs_step_2[0] == 1 or lbfgs_step_2[0] % 20 == 0: print(f"[Nonlinear L-BFGS] Step {lbfgs_step_2[0]:05d} | PDE = {loss.item():.6e}")
        return loss
    lbfgs2.step(closure_nonlinear)
    material.set_nonlinear_alpha(1.0)
    return model, history, source, material

def plot_results(model, source, material, history, resolution=160):
    fields = evaluate_fields(model, source, material, resolution=resolution, device=DEVICE)
    X, Y, Az, Bx, By, Bmag = fields["X"], fields["Y"], fields["Az"], fields["Bx"], fields["By"], fields["Bmag"]
    Mx, dMx_dy, rho_m, Mur = fields["Mx"], fields["dMx_dy"], fields["rho_m"], fields["Mur"]
    magnet_mask, iron_mask, Hmag = fields["magnet_mask"], fields["iron_mask"], fields["Hmag"]

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Az, levels=60, cmap="jet")
    plt.colorbar(contour, label="A_z (Wb/m)")
    plt.title("Magnetic Vector Potential $A_z$")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Bmag, levels=60, cmap="jet")
    plt.colorbar(contour, label="|B| (T)")
    plt.title("Magnetic Flux Density $|B|$")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    step, scale = max(1, resolution // 25), np.maximum(np.sqrt(Bx[::max(1, resolution // 25), ::max(1, resolution // 25)]**2 + By[::max(1, resolution // 25), ::max(1, resolution // 25)]**2), 1.0e-12)
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Bmag, levels=40, cmap="jet", alpha=0.25)
    plt.quiver(X[::step, ::step], Y[::step, ::step], Bx[::step, ::step] / scale, By[::step, ::step] / scale, scale=25.0, pivot="mid")
    plt.title("Magnetic Flux Density Vector $\\mathbf{B}$")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(material.iron_data.B_H_curve["H_data"], material.iron_data.B_H_curve["B_data"], "o-", linewidth=2, markersize=4, label="B-H data")
    B_op, H_op = Bmag[iron_mask > 0.95], Hmag[iron_mask > 0.95]
    if B_op.size > 0:
        idx = np.linspace(0, B_op.size - 1, min(3000, B_op.size), dtype=int)
        plt.scatter(H_op[idx], B_op[idx], s=3, alpha=0.20, label="PINN operating points")
    plt.xlabel("H (A/m)"); plt.ylabel("B (T)"); plt.title(f"B-H: {material.iron_data.name}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Mx, levels=60, cmap="viridis")
    plt.colorbar(contour, label="$M_x$ (A/m)")
    plt.title("Magnetization $M_x$")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, dMx_dy, levels=60, cmap="coolwarm")
    plt.colorbar(contour, label=r"$\partial M_x/\partial y$ (A/m$^2$)")
    plt.title("Magnetization Source Term")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    material_map = np.zeros_like(X); material_map[iron_mask > 0.5] = 2.0; material_map[(magnet_mask > 0.1) & ~(iron_mask > 0.5)] = 1.0
    contour = plt.contourf(X, Y, material_map, levels=[-0.5, 0.5, 1.5, 2.5], cmap="tab10")
    cbar = plt.colorbar(contour, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Air", "Magnets", "Iron Yoke"])
    plt.title("Material Map")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, rho_m, levels=60, cmap="bwr")
    plt.colorbar(contour, label=r"$\rho_m=-\partial M_x/\partial x$ (A/m$^2$)")
    plt.title("Magnetic Charge Density")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Mur, levels=60, cmap="copper")
    plt.colorbar(contour, label=r"$\mu_r(|B|)$")
    plt.title("Relative Permeability")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Hmag, levels=60, cmap="plasma")
    plt.colorbar(contour, label="H (A/m)")
    plt.title("Constitutive H Magnitude")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis("equal"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    adam_lin, lbfgs_lin, adam_nonlin, lbfgs_nonlin = np.asarray(history["adam_linear"]), np.asarray(history["lbfgs_linear"]), np.asarray(history["adam_nonlinear"]), np.asarray(history["lbfgs_nonlinear"])
    cursor = 0
    if adam_lin.size > 0: plt.semilogy(np.arange(1, len(adam_lin) + 1), np.maximum(adam_lin, 1.0e-20), label="Adam - Linear"); cursor += len(adam_lin)
    if lbfgs_lin.size > 0: plt.semilogy(np.arange(cursor + 1, cursor + len(lbfgs_lin) + 1), np.maximum(lbfgs_lin, 1.0e-20), label="L-BFGS - Linear"); cursor += len(lbfgs_lin)
    if adam_nonlin.size > 0: plt.semilogy(np.arange(cursor + 1, cursor + len(adam_nonlin) + 1), np.maximum(adam_nonlin, 1.0e-20), label="Adam - Nonlinear"); cursor += len(adam_nonlin)
    if lbfgs_nonlin.size > 0: plt.semilogy(np.arange(cursor + 1, cursor + len(lbfgs_nonlin) + 1), np.maximum(lbfgs_nonlin, 1.0e-20), label="L-BFGS - Nonlinear")
    plt.title("PDE Training History"); plt.xlabel("Training evaluation"); plt.ylabel("PDE loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    alpha_hist = np.asarray(history["alpha"])
    if alpha_hist.size > 0: plt.plot(np.arange(1, len(alpha_hist) + 1), alpha_hist, linewidth=2)
    plt.title("Nonlinear Continuation Parameter"); plt.xlabel("Nonlinear epoch"); plt.ylabel(r"$\alpha$"); plt.ylim(-0.05, 1.05); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    print("\n========================================\n        MAGNETIC PINN CONFIGURATION\n========================================")
    print(f"Material               : {IRON_MATERIAL_NAME}\nLeft polarity          : {LEFT_POLARITY:+.1f}\nRight polarity         : {RIGHT_POLARITY:+.1f}\nDomain                 : [-{DOMAIN_EXTENT}, {DOMAIN_EXTENT}]^2\nM0                     : {M0:.6e} A/m\nB0                     : {B0:.6e} T\n========================================")
    trained_model, history, source, material = train_pinn_curriculum(epochs_linear=3000, lbfgs_linear=150, epochs_nonlinear=1800, lbfgs_nonlinear=100, num_collocation=10000, lbfgs_collocation=5000, hidden_layers=6, neurons=128, linear_lr=1.0e-3, nonlinear_lr=2.0e-4, nonlinear_ramp_fraction=0.40, print_every_linear=250, print_every_nonlinear=100, grad_clip_linear=1.0, grad_clip_nonlinear=0.5, adaptive_refinement=True, adaptive_interval=500, adaptive_ratio=0.05)
    report_diagnostics(trained_model, source, material, num_validation=5000)
    plot_results(trained_model, source, material, history, resolution=160)