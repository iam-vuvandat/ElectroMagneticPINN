import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

MU0 = 4.0 * math.pi * 1.0e-7
MUR_MAGNET = 1.05
M0 = 1.0e6
L0 = 1.0
A0 = MU0 * M0 * L0
B0 = MU0 * M0

DOMAIN_EXTENT = 1.2
LENGTH_X = 0.30
WIDTH_Y = 0.12
RADIUS = 0.03
CENTER_OFFSET_X = 0.40

POLARITY_L = 1.0
POLARITY_R = 1.0

def rounded_box_sdf(xy, center_x, center_y, half_w, half_h, r):
    x = xy[:, 0:1] - center_x
    y = xy[:, 1:2] - center_y
    q_x = torch.abs(x) - (half_w - r)
    q_y = torch.abs(y) - (half_h - r)
    q_max_x = torch.maximum(q_x, torch.zeros_like(q_x))
    q_max_y = torch.maximum(q_y, torch.zeros_like(q_y))
    dist_out = torch.sqrt(q_max_x**2 + q_max_y**2)
    dist_in = torch.minimum(torch.maximum(q_x, q_y), torch.zeros_like(q_x))
    return dist_out + dist_in - r

class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, sigma):
        super().__init__()
        self.register_buffer('B', torch.randn(in_features, mapping_size) * sigma)
        
    def forward(self, x):
        x_proj = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SubNet(nn.Module):
    def __init__(self, hidden_layers=8, neurons=128, fourier_mapping_size=256, fourier_sigma=1.5):
        super().__init__()
        self.fourier = FourierFeatures(2, fourier_mapping_size, fourier_sigma)
        
        layers = [nn.Linear(fourier_mapping_size * 2, neurons), nn.SiLU()]
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

    def forward(self, xy):
        ff_out = self.fourier(xy)
        return self.net(ff_out)

class MagneticXPINN(nn.Module):
    def __init__(self, hidden_layers=8, neurons=128):
        super().__init__()
        self.net_air = SubNet(hidden_layers, neurons, fourier_mapping_size=256, fourier_sigma=1.5)
        self.net_mag_L = SubNet(hidden_layers, neurons, fourier_mapping_size=256, fourier_sigma=1.5)
        self.net_mag_R = SubNet(hidden_layers, neurons, fourier_mapping_size=256, fourier_sigma=1.5)

    def boundary_factor(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        return (1.0 - (x / DOMAIN_EXTENT)**2) * (1.0 - (y / DOMAIN_EXTENT)**2)

    def forward_air(self, xy):
        return self.boundary_factor(xy) * self.net_air(xy)

    def forward_mag_L(self, xy):
        return self.net_mag_L(xy)
        
    def forward_mag_R(self, xy):
        return self.net_mag_R(xy)

def create_sobol_engine(seed=SEED):
    return torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)

def sample_projected_interface(num_points, engine, center_x, center_y, device=DEVICE):
    uv = engine.draw(num_points * 6).to(device=device, dtype=DTYPE)
    xy_rand = (-DOMAIN_EXTENT + (2.0 * DOMAIN_EXTENT) * uv).requires_grad_(True)
    
    sdf_rand = rounded_box_sdf(xy_rand, center_x, center_y, LENGTH_X, WIDTH_Y, RADIUS)
    mask_close = (sdf_rand.squeeze() > -0.15) & (sdf_rand.squeeze() < 0.15)
    xy_close = xy_rand[mask_close]
    
    if xy_close.shape[0] > num_points:
        xy_close = xy_close[:num_points]
        
    sdf_close = rounded_box_sdf(xy_close, center_x, center_y, LENGTH_X, WIDTH_Y, RADIUS)
    grad_close = torch.autograd.grad(sdf_close, xy_close, grad_outputs=torch.ones_like(sdf_close))[0]
    
    xy_I = (xy_close - sdf_close * grad_close).detach().requires_grad_(True)
    return xy_I

def sample_xpinn_points(num_points, engine, device=DEVICE):
    uv = engine.draw(num_points).to(device=device, dtype=DTYPE)
    xy_all = (-DOMAIN_EXTENT + (2.0 * DOMAIN_EXTENT) * uv).requires_grad_(True)
    
    sdf_L = rounded_box_sdf(xy_all, -CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
    sdf_R = rounded_box_sdf(xy_all, CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
    
    mask_mag_L = (sdf_L <= 0.0).squeeze()
    mask_mag_R = (sdf_R <= 0.0).squeeze()
    mask_air = ~(mask_mag_L | mask_mag_R)
    
    xy_mag_L = xy_all[mask_mag_L].detach().requires_grad_(True)
    xy_mag_R = xy_all[mask_mag_R].detach().requires_grad_(True)
    xy_air_uniform = xy_all[mask_air].detach()
    
    uv_gap = engine.draw(num_points // 2).to(device=device, dtype=DTYPE)
    x_gap = -0.5 + 1.0 * uv_gap[:, 0:1]
    y_gap = -0.2 + 0.4 * uv_gap[:, 1:2]
    xy_gap = torch.cat([x_gap, y_gap], dim=1)
    
    sdf_gap_L = rounded_box_sdf(xy_gap, -CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
    sdf_gap_R = rounded_box_sdf(xy_gap, CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
    mask_gap_air = (sdf_gap_L > 0.0) & (sdf_gap_R > 0.0)
    xy_gap_air = xy_gap[mask_gap_air.squeeze()].detach()
    
    xy_air = torch.cat([xy_air_uniform, xy_gap_air], dim=0).requires_grad_(True)
    
    num_I = num_points // 4
    xy_I_L = sample_projected_interface(num_I, engine, -CENTER_OFFSET_X, 0.0, device)
    xy_I_R = sample_projected_interface(num_I, engine, CENTER_OFFSET_X, 0.0, device)
    
    return xy_mag_L, xy_mag_R, xy_air, xy_I_L, xy_I_R

def compute_gradients(u, x):
    return torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def compute_domain_loss(model_fn, xy, mx_val, center_x_mag=None):
    if xy.shape[0] == 0:
        return torch.tensor(0.0, device=DEVICE)
    Az = model_fn(xy)
    grad = compute_gradients(Az, xy)
    Bx = grad[:, 1:2]
    By = -grad[:, 0:1]
    
    # Tính khoảng cách SDF để tạo trọng số làm mờ biên
    if mx_val == 0.0:
        Hx = 1.0 * Bx
        Hy = 1.0 * By
        sdf_L = rounded_box_sdf(xy, -CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
        sdf_R = rounded_box_sdf(xy, CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
        abs_sdf = torch.minimum(torch.abs(sdf_L), torch.abs(sdf_R))
    else:
        nu_mag = 1.0 / MUR_MAGNET
        Hx = nu_mag * Bx - mx_val
        Hy = nu_mag * By
        abs_sdf = torch.abs(rounded_box_sdf(xy, center_x_mag, 0.0, LENGTH_X, WIDTH_Y, RADIUS))
        
    dHx_dy = compute_gradients(Hx, xy)[:, 1:2]
    dHy_dx = compute_gradients(Hy, xy)[:, 0:1]
    sq_res = (dHy_dx - dHx_dy)**2
    
    # Kỹ thuật PDE Boundary Fading: dập loss về 0 khi tiến sát ranh giới
    weight = 1.0 - torch.exp(-150.0 * abs_sdf)
    return torch.mean(sq_res * weight)

def compute_interface_loss(model_air_fn, model_mag_fn, xy_I, center_x, center_y, mx_val):
    if xy_I.shape[0] == 0:
        return torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)
        
    sdf_I = rounded_box_sdf(xy_I, center_x, center_y, LENGTH_X, WIDTH_Y, RADIUS)
    n_vec = torch.autograd.grad(sdf_I, xy_I, grad_outputs=torch.ones_like(sdf_I), create_graph=True)[0]
    n_norm = torch.linalg.norm(n_vec, dim=1, keepdim=True)
    n_vec = n_vec / (n_norm + 1e-8)
    nx = n_vec[:, 0:1]
    ny = n_vec[:, 1:2]
    tx = -ny
    ty = nx

    Az_air = model_air_fn(xy_I)
    Az_mag = model_mag_fn(xy_I)
    loss_val = torch.mean((Az_air - Az_mag)**2)
    
    grad_air = compute_gradients(Az_air, xy_I)
    grad_mag = compute_gradients(Az_mag, xy_I)
    
    Bx_air = grad_air[:, 1:2]
    By_air = -grad_air[:, 0:1]
    Bx_mag = grad_mag[:, 1:2]
    By_mag = -grad_mag[:, 0:1]
    
    Hx_air = 1.0 * Bx_air
    Hy_air = 1.0 * By_air
    
    nu_mag = 1.0 / MUR_MAGNET
    Hx_mag = nu_mag * Bx_mag - mx_val
    Hy_mag = nu_mag * By_mag
    
    Bn_air = Bx_air * nx + By_air * ny
    Bn_mag = Bx_mag * nx + By_mag * ny
    loss_Bn = torch.mean((Bn_air - Bn_mag)**2)
    
    Ht_air = Hx_air * tx + Hy_air * ty
    Ht_mag = Hx_mag * tx + Hy_mag * ty
    loss_Ht = torch.mean((Ht_air - Ht_mag)**2)
    
    return loss_val, loss_Bn, loss_Ht

def compute_xpinn_loss(model, xy_mag_L, xy_mag_R, xy_air, xy_I_L, xy_I_R):
    loss_pde_air = compute_domain_loss(model.forward_air, xy_air, 0.0)
    loss_pde_L = compute_domain_loss(model.forward_mag_L, xy_mag_L, POLARITY_L, -CENTER_OFFSET_X)
    loss_pde_R = compute_domain_loss(model.forward_mag_R, xy_mag_R, POLARITY_R, CENTER_OFFSET_X)
    
    val_L, Bn_L, Ht_L = compute_interface_loss(model.forward_air, model.forward_mag_L, xy_I_L, -CENTER_OFFSET_X, 0.0, POLARITY_L)
    val_R, Bn_R, Ht_R = compute_interface_loss(model.forward_air, model.forward_mag_R, xy_I_R, CENTER_OFFSET_X, 0.0, POLARITY_R)
    
    loss_val = val_L + val_R
    loss_B_n = Bn_L + Bn_R
    loss_H_t = Ht_L + Ht_R

    loss_total = loss_pde_air + loss_pde_L + loss_pde_R + 1000.0 * loss_val + 500.0 * loss_H_t + 500.0 * loss_B_n
    return loss_total

def train_xpinn(epochs=6000, num_collocation=20000, hidden_layers=8, neurons=128, learning_rate=1.0e-3, lbfgs_max_iter=3000):
    model = MagneticXPINN(hidden_layers=hidden_layers, neurons=neurons).to(DEVICE)
    adam = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=epochs, eta_min=1.0e-6)
    engine = create_sobol_engine(seed=SEED)

    history = {"adam_epoch": [], "adam_loss": [], "lbfgs_step": [], "lbfgs_loss": []}
    model.train()

    print("\n========================================")
    print("                  ADAM")
    print("========================================")

    for epoch in range(1, epochs + 1):
        if epoch == 1 or epoch % 100 == 0:
            xy_mag_L, xy_mag_R, xy_air, xy_I_L, xy_I_R = sample_xpinn_points(num_collocation, engine)

        adam.zero_grad(set_to_none=True)
        loss = compute_xpinn_loss(model, xy_mag_L, xy_mag_R, xy_air, xy_I_L, xy_I_R)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adam.step()
        scheduler.step()

        history["adam_epoch"].append(epoch)
        history["adam_loss"].append(loss.item())

        if epoch == 1 or epoch % 250 == 0:
            print(f"Epoch {epoch:05d} | Total Loss = {loss.item():.6e}")

    print("\n========================================")
    print("                 L-BFGS")
    print("========================================")

    lbfgs_engine = create_sobol_engine(seed=SEED + 1)
    xy_mag_L, xy_mag_R, xy_air, xy_I_L, xy_I_R = sample_xpinn_points(num_collocation, lbfgs_engine)
    
    lbfgs = optim.LBFGS(
        model.parameters(), lr=0.8, max_iter=lbfgs_max_iter, max_eval=int(lbfgs_max_iter * 1.25),
        history_size=50, tolerance_grad=1.0e-8, tolerance_change=1.0e-10, line_search_fn="strong_wolfe"
    )

    lbfgs_counter = [0]

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss = compute_xpinn_loss(model, xy_mag_L, xy_mag_R, xy_air, xy_I_L, xy_I_R)
        loss.backward()
        
        lbfgs_counter[0] += 1
        history["lbfgs_step"].append(lbfgs_counter[0])
        history["lbfgs_loss"].append(loss.item())

        if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
            print(f"Step {lbfgs_counter[0]:05d} | Total Loss = {loss.item():.6e}")
        return loss

    lbfgs.step(closure)
    return model, history

def evaluate_fields(model, resolution=160):
    model.eval()
    x_star = np.linspace(-DOMAIN_EXTENT, DOMAIN_EXTENT, resolution, dtype=np.float32)
    y_star = np.linspace(-DOMAIN_EXTENT, DOMAIN_EXTENT, resolution, dtype=np.float32)
    X_star, Y_star = np.meshgrid(x_star, y_star)
    
    xy_np = np.column_stack((X_star.ravel(), Y_star.ravel())).astype(np.float32)
    xy = torch.tensor(xy_np, device=DEVICE, dtype=DTYPE, requires_grad=True)
    
    sdf_L = rounded_box_sdf(xy, -CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
    sdf_R = rounded_box_sdf(xy, CENTER_OFFSET_X, 0.0, LENGTH_X, WIDTH_Y, RADIUS)
    
    mask_L = (sdf_L <= 0.0)
    mask_R = (sdf_R <= 0.0)
    
    Az_air = model.forward_air(xy)
    Az_mag_L = model.forward_mag_L(xy)
    Az_mag_R = model.forward_mag_R(xy)
    
    grad_air = compute_gradients(Az_air, xy)
    grad_L = compute_gradients(Az_mag_L, xy)
    grad_R = compute_gradients(Az_mag_R, xy)
    
    Bx_air = grad_air[:, 1:2]
    By_air = -grad_air[:, 0:1]
    Bx_L = grad_L[:, 1:2]
    By_L = -grad_L[:, 0:1]
    Bx_R = grad_R[:, 1:2]
    By_R = -grad_R[:, 0:1]
    
    dHx_dy_air = compute_gradients(1.0 * Bx_air, xy)[:, 1:2]
    dHy_dx_air = compute_gradients(1.0 * By_air, xy)[:, 0:1]
    res_air = (dHy_dx_air - dHx_dy_air)**2

    nu_mag = 1.0 / MUR_MAGNET
    dHx_dy_L = compute_gradients(nu_mag * Bx_L - POLARITY_L, xy)[:, 1:2]
    dHy_dx_L = compute_gradients(nu_mag * By_L, xy)[:, 0:1]
    res_L = (dHy_dx_L - dHx_dy_L)**2

    dHx_dy_R = compute_gradients(nu_mag * Bx_R - POLARITY_R, xy)[:, 1:2]
    dHy_dx_R = compute_gradients(nu_mag * By_R, xy)[:, 0:1]
    res_R = (dHy_dx_R - dHx_dy_R)**2

    res_star_sq = torch.where(mask_L, res_L, torch.where(mask_R, res_R, res_air))
    
    # Tính sai số thực tế không bọc biên để hiển thị trên đồ thị
    res_star = torch.sqrt(res_star_sq)
    Residual = res_star.detach().cpu().numpy().reshape(X_star.shape)

    Bx_star = torch.where(mask_L, Bx_L, torch.where(mask_R, Bx_R, Bx_air))
    By_star = torch.where(mask_L, By_L, torch.where(mask_R, By_R, By_air))
    
    Bx = (B0 * Bx_star).detach().cpu().numpy().reshape(X_star.shape)
    By = (B0 * By_star).detach().cpu().numpy().reshape(X_star.shape)
    Bmag = np.sqrt(Bx**2 + By**2)
    
    return {"X": L0 * X_star, "Y": L0 * Y_star, "Bx": Bx, "By": By, "Bmag": Bmag, "Residual": Residual}

def plot_results(model, history, resolution=160):
    fields = evaluate_fields(model, resolution=resolution)
    X, Y = fields["X"], fields["Y"]
    Bx, By, Bmag, Residual = fields["Bx"], fields["By"], fields["Bmag"], fields["Residual"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    adam_loss = np.asarray(history["adam_loss"])
    lbfgs_loss = np.asarray(history["lbfgs_loss"])
    if adam_loss.size > 0:
        axes[0].semilogy(np.arange(1, adam_loss.size + 1), np.maximum(adam_loss, 1.0e-20), label="Adam", color='blue')
    if lbfgs_loss.size > 0:
        start = adam_loss.size + 1
        axes[0].semilogy(np.arange(start, start + lbfgs_loss.size), np.maximum(lbfgs_loss, 1.0e-20), label="L-BFGS", color='red')
    axes[0].set_title("XPINN Training Loss History (2 Magnets)")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    levels_B = np.linspace(0.0, 2.0, 17)
    contour = axes[1].contourf(X, Y, Bmag, levels=levels_B, cmap="jet", extend="max")
    fig.colorbar(contour, ax=axes[1], label="|B| (T)")
    axes[1].set_title("Magnetic Flux Density Magnitude $|B|$")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")

    step = 12 
    Bsub = np.sqrt(Bx[::step, ::step]**2 + By[::step, ::step]**2)
    scale = np.maximum(Bsub, 1.0e-12)
    axes[2].contourf(X, Y, Bmag, levels=levels_B, cmap="jet", alpha=0.25, extend="max")
    axes[2].quiver(X[::step, ::step], Y[::step, ::step], 
                   Bx[::step, ::step]/scale, By[::step, ::step]/scale, 
                   scale=40.0, pivot="mid")
    axes[2].set_title("Magnetic Flux Density Vector $\\mathbf{B}$")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")

    res_abs = np.abs(Residual) + 1e-8
    levels = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    contour_res = axes[3].contourf(X, Y, res_abs, levels=levels, cmap="Reds", 
                                   norm=LogNorm(vmin=1e-4, vmax=1e1), extend='both')
    cbar = fig.colorbar(contour_res, ax=axes[3])
    cbar.set_ticks(levels)
    axes[3].set_title("PDE Residual Error")
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].set_xlabel("x (m)")
    axes[3].set_ylabel("y (m)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, history = train_xpinn(
        epochs=6000, num_collocation=10000, hidden_layers=6, neurons=128, learning_rate=1.0e-3, lbfgs_max_iter=3000
    )
    
    plot_results(trained_model, history, resolution=160)