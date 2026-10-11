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

MU0 = 4.0 * math.pi * 1.0e-7
MUR_MAGNET = 1.05
M0 = 1.0e6
L0 = 1.0
A0 = MU0 * M0 * L0
B0 = MU0 * M0

DOMAIN_EXTENT = 2.0
LENGTH_X = 0.30
WIDTH_Y = 0.12

class SubNet(nn.Module):
    def __init__(self, hidden_layers=6, neurons=128):
        super().__init__()
        layers = [nn.Linear(2, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy):
        return self.net(xy)

class MagneticXPINN(nn.Module):
    def __init__(self, hidden_layers=6, neurons=128):
        super().__init__()
        self.net_air = SubNet(hidden_layers, neurons)
        self.net_mag = SubNet(hidden_layers, neurons)

    def boundary_factor(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        return (1.0 - (x / DOMAIN_EXTENT)**2) * (1.0 - (y / DOMAIN_EXTENT)**2)

    def forward_air(self, xy):
        return self.boundary_factor(xy) * self.net_air(xy)

    def forward_mag(self, xy):
        return self.net_mag(xy)

def create_sobol_engine(seed=SEED):
    return torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)

def sample_xpinn_points(num_points, engine, device=DEVICE):
    uv = engine.draw(num_points).to(device=device, dtype=DTYPE)
    x = -DOMAIN_EXTENT + (2.0 * DOMAIN_EXTENT) * uv[:, 0:1]
    y = -DOMAIN_EXTENT + (2.0 * DOMAIN_EXTENT) * uv[:, 1:2]
    
    mask_mag = (x >= -LENGTH_X) & (x <= LENGTH_X) & (y >= -WIDTH_Y) & (y <= WIDTH_Y)
    xy_all = torch.cat([x, y], dim=1)
    
    xy_mag = xy_all[mask_mag.squeeze()].requires_grad_(True)
    xy_air = xy_all[~mask_mag.squeeze()].requires_grad_(True)
    
    num_I = num_points // 10
    uv_I = engine.draw(num_I).to(device=device, dtype=DTYPE)
    
    x_I = -LENGTH_X + (2.0 * LENGTH_X) * uv_I[:, 0:1]
    I_top = torch.cat([x_I, torch.full_like(x_I, WIDTH_Y)], dim=1).requires_grad_(True)
    I_bot = torch.cat([x_I, torch.full_like(x_I, -WIDTH_Y)], dim=1).requires_grad_(True)
    
    y_I = -WIDTH_Y + (2.0 * WIDTH_Y) * uv_I[:, 1:2]
    I_left = torch.cat([torch.full_like(y_I, -LENGTH_X), y_I], dim=1).requires_grad_(True)
    I_right = torch.cat([torch.full_like(y_I, LENGTH_X), y_I], dim=1).requires_grad_(True)
    
    return xy_mag, xy_air, I_top, I_bot, I_left, I_right

def compute_gradients(u, x):
    return torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def compute_xpinn_loss(model, xy_mag, xy_air, I_top, I_bot, I_left, I_right):
    if xy_air.shape[0] > 0:
        Az_air = model.forward_air(xy_air)
        grad_air = compute_gradients(Az_air, xy_air)
        Bx_air = grad_air[:, 1:2]
        By_air = -grad_air[:, 0:1]
        Hx_air = 1.0 * Bx_air
        Hy_air = 1.0 * By_air
        dHx_dy_air = compute_gradients(Hx_air, xy_air)[:, 1:2]
        dHy_dx_air = compute_gradients(Hy_air, xy_air)[:, 0:1]
        loss_pde_air = torch.mean((dHy_dx_air - dHx_dy_air)**2)
    else:
        loss_pde_air = torch.tensor(0.0, device=DEVICE)

    if xy_mag.shape[0] > 0:
        Az_mag = model.forward_mag(xy_mag)
        grad_mag = compute_gradients(Az_mag, xy_mag)
        Bx_mag = grad_mag[:, 1:2]
        By_mag = -grad_mag[:, 0:1]
        nu_mag = 1.0 / MUR_MAGNET
        Mx = 1.0
        Hx_mag = nu_mag * Bx_mag - Mx
        Hy_mag = nu_mag * By_mag
        dHx_dy_mag = compute_gradients(Hx_mag, xy_mag)[:, 1:2]
        dHy_dx_mag = compute_gradients(Hy_mag, xy_mag)[:, 0:1]
        loss_pde_mag = torch.mean((dHy_dx_mag - dHx_dy_mag)**2)
    else:
        loss_pde_mag = torch.tensor(0.0, device=DEVICE)

    loss_val = torch.tensor(0.0, device=DEVICE)
    loss_flux = torch.tensor(0.0, device=DEVICE)
    nu_mag = 1.0 / MUR_MAGNET
    Mx = 1.0

    for I_pts, is_horizontal in [(I_top, True), (I_bot, True), (I_left, False), (I_right, False)]:
        Az_I_air = model.forward_air(I_pts)
        Az_I_mag = model.forward_mag(I_pts)
        loss_val += torch.mean((Az_I_air - Az_I_mag)**2)
        
        grad_I_air = compute_gradients(Az_I_air, I_pts)
        grad_I_mag = compute_gradients(Az_I_mag, I_pts)
        
        Hx_I_air = 1.0 * grad_I_air[:, 1:2]
        Hy_I_air = -1.0 * grad_I_air[:, 0:1]
        Hx_I_mag = nu_mag * grad_I_mag[:, 1:2] - Mx
        Hy_I_mag = -nu_mag * grad_I_mag[:, 0:1]
        
        if is_horizontal:
            loss_flux += torch.mean((Hx_I_air - Hx_I_mag)**2)
        else:
            loss_flux += torch.mean((Hy_I_air - Hy_I_mag)**2)

    loss_total = loss_pde_air + loss_pde_mag + loss_val + loss_flux
    return loss_total, loss_pde_air, loss_pde_mag, loss_val, loss_flux

def train_xpinn(epochs=4000, num_collocation=10000, hidden_layers=6, neurons=128, learning_rate=1.0e-3, lbfgs_max_iter=1000):
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
        adam.zero_grad(set_to_none=True)
        xy_mag, xy_air, I_top, I_bot, I_left, I_right = sample_xpinn_points(num_collocation, engine)
        
        loss, l_air, l_mag, l_val, l_flux = compute_xpinn_loss(model, xy_mag, xy_air, I_top, I_bot, I_left, I_right)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adam.step()
        scheduler.step()

        history["adam_epoch"].append(epoch)
        history["adam_loss"].append(loss.item())

        if epoch == 1 or epoch % 250 == 0:
            print(f"Epoch {epoch:05d} | Loss = {loss.item():.6e} | Val = {l_val.item():.6e} | Flux = {l_flux.item():.6e}")

    print("\n========================================")
    print("                 L-BFGS")
    print("========================================")

    lbfgs_engine = create_sobol_engine(seed=SEED + 1)
    xy_mag, xy_air, I_top, I_bot, I_left, I_right = sample_xpinn_points(num_collocation, lbfgs_engine)
    
    lbfgs = optim.LBFGS(
        model.parameters(), lr=0.8, max_iter=lbfgs_max_iter, max_eval=1250,
        history_size=50, tolerance_grad=1.0e-8, tolerance_change=1.0e-10, line_search_fn="strong_wolfe"
    )

    lbfgs_counter = [0]

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss, _, _, _, _ = compute_xpinn_loss(model, xy_mag, xy_air, I_top, I_bot, I_left, I_right)
        loss.backward()
        
        lbfgs_counter[0] += 1
        history["lbfgs_step"].append(lbfgs_counter[0])
        history["lbfgs_loss"].append(loss.item())

        if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
            print(f"Step {lbfgs_counter[0]:05d} | Loss = {loss.item():.6e}")
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
    
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    mask_mag = (x >= -LENGTH_X) & (x <= LENGTH_X) & (y >= -WIDTH_Y) & (y <= WIDTH_Y)
    
    Az_air = model.forward_air(xy)
    Az_mag = model.forward_mag(xy)
    Az_star = torch.where(mask_mag, Az_mag, Az_air)
    
    grad_air = compute_gradients(Az_air, xy)
    grad_mag = compute_gradients(Az_mag, xy)
    
    Bx_air = grad_air[:, 1:2]
    By_air = -grad_air[:, 0:1]
    Bx_mag = grad_mag[:, 1:2]
    By_mag = -grad_mag[:, 0:1]
    
    Bx_star = torch.where(mask_mag, Bx_mag, Bx_air)
    By_star = torch.where(mask_mag, By_mag, By_air)
    
    Az = (A0 * Az_star).detach().cpu().numpy().reshape(X_star.shape)
    Bx = (B0 * Bx_star).detach().cpu().numpy().reshape(X_star.shape)
    By = (B0 * By_star).detach().cpu().numpy().reshape(X_star.shape)
    Bmag = np.sqrt(Bx**2 + By**2)
    
    Mur_np = np.where((X_star >= -LENGTH_X) & (X_star <= LENGTH_X) & (Y_star >= -WIDTH_Y) & (Y_star <= WIDTH_Y), MUR_MAGNET, 1.0)
    Mx_np = np.where((X_star >= -LENGTH_X) & (X_star <= LENGTH_X) & (Y_star >= -WIDTH_Y) & (Y_star <= WIDTH_Y), M0, 0.0)
    
    return {"X": L0 * X_star, "Y": L0 * Y_star, "Az": Az, "Bx": Bx, "By": By, "Bmag": Bmag, "Mur": Mur_np, "Mx": Mx_np}

def plot_results(model, history, resolution=160):
    fields = evaluate_fields(model, resolution=resolution)
    X, Y = fields["X"], fields["Y"]
    Az, Bmag, Bx, By = fields["Az"], fields["Bmag"], fields["Bx"], fields["By"]
    Mur, Mx = fields["Mur"], fields["Mx"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    contour = axes[0, 0].contourf(X, Y, Az, levels=60, cmap="jet")
    fig.colorbar(contour, ax=axes[0, 0], label="A_z (Wb/m)")
    axes[0, 0].set_title("Magnetic Vector Potential $A_z$")

    contour = axes[0, 1].contourf(X, Y, Bmag, levels=60, cmap="jet")
    fig.colorbar(contour, ax=axes[0, 1], label="|B| (T)")
    axes[0, 1].set_title("Magnetic Flux Density $|B|$")

    contour = axes[0, 2].contourf(X, Y, Mx, levels=60, cmap="viridis")
    fig.colorbar(contour, ax=axes[0, 2], label="$M_x$ (A/m)")
    axes[0, 2].set_title("Magnetization $M_x$")

    step = max(1, resolution // 25)
    Bsub = np.sqrt(Bx[::step, ::step]**2 + By[::step, ::step]**2)
    scale = np.maximum(Bsub, 1.0e-12)
    axes[1, 0].contourf(X, Y, Bmag, levels=40, cmap="jet", alpha=0.25)
    axes[1, 0].quiver(X[::step, ::step], Y[::step, ::step], Bx[::step, ::step]/scale, By[::step, ::step]/scale, scale=25.0, pivot="mid")
    axes[1, 0].set_title("Magnetic Flux Density Vector $\\mathbf{B}$")

    contour = axes[1, 1].contourf(X, Y, Mur, levels=60, cmap="copper")
    fig.colorbar(contour, ax=axes[1, 1], label=r"$\mu_r$")
    axes[1, 1].set_title("Relative Permeability $\\mu_r$")

    adam_loss = np.asarray(history["adam_loss"])
    lbfgs_loss = np.asarray(history["lbfgs_loss"])
    if adam_loss.size > 0:
        axes[1, 2].semilogy(np.arange(1, adam_loss.size + 1), np.maximum(adam_loss, 1.0e-20), label="Adam")
    if lbfgs_loss.size > 0:
        start = adam_loss.size + 1
        axes[1, 2].semilogy(np.arange(start, start + lbfgs_loss.size), np.maximum(lbfgs_loss, 1.0e-20), label="L-BFGS")
    axes[1, 2].set_title("XPINN Training History")
    axes[1, 2].set_xlabel("Training evaluation")
    axes[1, 2].set_ylabel("Total Loss")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat[:-1]:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    plt.tight_layout()
    plt.savefig('xpinn_straight_magnet.png')
    print("\n[*] Hoàn thành. Kết quả biểu đồ được lưu tại 'xpinn_straight_magnet.png'")

if __name__ == "__main__":
    trained_model, history = train_xpinn(
        epochs=4000, num_collocation=10000, hidden_layers=6, neurons=128, learning_rate=1.0e-3, lbfgs_max_iter=1000
    )
    plot_results(trained_model, history, resolution=160)