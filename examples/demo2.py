import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# Global configuration
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DTYPE = torch.float32

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    print(
        f"[DEBUG] Device: GPU "
        f"({torch.cuda.get_device_name(0)})"
    )
else:
    print("[DEBUG] Device: CPU")


# ============================================================
# Physical constants / normalization
# ============================================================

MU0 = 4.0 * math.pi * 1.0e-7

MUR_IRON = 400.0

M0 = 1.0e6

L0 = 1.0

A0 = MU0 * M0 * L0
B0 = MU0 * M0


# ============================================================
# Geometry
# ============================================================

DOMAIN_EXTENT = 2.0

LENGTH_X = 0.30
WIDTH_Y = 0.12
STEEPNESS = 60.0

MAGNET_OFFSET_X = 1.1 * LENGTH_X

SPAN_X = MAGNET_OFFSET_X + LENGTH_X

YOKE_LENGTH_X = 1.10 * SPAN_X
YOKE_WIDTH_Y = WIDTH_Y
YOKE_OFFSET_Y = -(2.0 * WIDTH_Y)


# ============================================================
# Magnet polarity
# ============================================================

# Current configuration:
#
# Original:
# LEFT  = -1
# RIGHT = +1
#
# Here the LEFT magnet is reversed:
# LEFT  = +1
# RIGHT = +1
#
# Other examples:
#
# Opposite polarity:
# LEFT_POLARITY  = -1.0
# RIGHT_POLARITY = +1.0
#
# Reverse right magnet:
# LEFT_POLARITY  = -1.0
# RIGHT_POLARITY = -1.0
#
# Reverse both:
# LEFT_POLARITY  = +1.0
# RIGHT_POLARITY = -1.0

LEFT_POLARITY = +1.0
RIGHT_POLARITY = +1.0


# ============================================================
# Magnetization source
# ============================================================

class MagnetizationSource:

    def __init__(
        self,
        length_x=LENGTH_X,
        width_y=WIDTH_Y,
        offset_x=MAGNET_OFFSET_X,
        steepness=STEEPNESS,
        left_polarity=LEFT_POLARITY,
        right_polarity=RIGHT_POLARITY
    ):
        self.lx = float(length_x)
        self.wy = float(width_y)
        self.offset_x = float(offset_x)
        self.k = float(steepness)

        self.left_polarity = float(left_polarity)
        self.right_polarity = float(right_polarity)

    def dimensionless(self, xy):

        x = xy[:, 0:1]
        y = xy[:, 1:2]

        sig_x_left = (
            torch.sigmoid(
                self.k * (
                    x
                    + self.offset_x
                    + self.lx
                )
            )
            -
            torch.sigmoid(
                self.k * (
                    x
                    + self.offset_x
                    - self.lx
                )
            )
        )

        sig_x_right = (
            torch.sigmoid(
                self.k * (
                    x
                    - self.offset_x
                    + self.lx
                )
            )
            -
            torch.sigmoid(
                self.k * (
                    x
                    - self.offset_x
                    - self.lx
                )
            )
        )

        sig_y = (
            torch.sigmoid(
                self.k * (y + self.wy)
            )
            -
            torch.sigmoid(
                self.k * (y - self.wy)
            )
        )

        return (
            self.left_polarity * sig_x_left
            +
            self.right_polarity * sig_x_right
        ) * sig_y

    def physical(self, xy):
        return (
            M0
            * self.dimensionless(xy)
        )

    def dmx_dy_dimensionless(self, xy):

        x = xy[:, 0:1]
        y = xy[:, 1:2]

        sig_x_left = (
            torch.sigmoid(
                self.k * (
                    x
                    + self.offset_x
                    + self.lx
                )
            )
            -
            torch.sigmoid(
                self.k * (
                    x
                    + self.offset_x
                    - self.lx
                )
            )
        )

        sig_x_right = (
            torch.sigmoid(
                self.k * (
                    x
                    - self.offset_x
                    + self.lx
                )
            )
            -
            torch.sigmoid(
                self.k * (
                    x
                    - self.offset_x
                    - self.lx
                )
            )
        )

        s_y_plus = torch.sigmoid(
            self.k * (y + self.wy)
        )

        s_y_minus = torch.sigmoid(
            self.k * (y - self.wy)
        )

        ds_y = (
            self.k
            * s_y_plus
            * (1.0 - s_y_plus)
            -
            self.k
            * s_y_minus
            * (1.0 - s_y_minus)
        )

        return (
            self.left_polarity * sig_x_left
            +
            self.right_polarity * sig_x_right
        ) * ds_y

    def dmx_dy_physical(self, xy):

        return (
            M0 / L0
        ) * self.dmx_dy_dimensionless(xy)


# ============================================================
# Material model
# ============================================================

class MaterialPermeability:

    def __init__(
        self,
        yoke_lx=YOKE_LENGTH_X,
        yoke_wy=YOKE_WIDTH_Y,
        yoke_offset_y=YOKE_OFFSET_Y,
        mur=MUR_IRON,
        steepness=STEEPNESS
    ):

        self.lx = float(yoke_lx)
        self.wy = float(yoke_wy)
        self.y0 = float(yoke_offset_y)

        self.mur = float(mur)
        self.k = float(steepness)

        self.nu_air = 1.0
        self.nu_iron = 1.0 / self.mur

    def iron_mask(self, xy):

        x = xy[:, 0:1]
        y = xy[:, 1:2]

        sig_x = (
            torch.sigmoid(
                self.k * (x + self.lx)
            )
            -
            torch.sigmoid(
                self.k * (x - self.lx)
            )
        )

        sig_y = (
            torch.sigmoid(
                self.k * (
                    y
                    - self.y0
                    + self.wy
                )
            )
            -
            torch.sigmoid(
                self.k * (
                    y
                    - self.y0
                    - self.wy
                )
            )
        )

        return sig_x * sig_y

    def nu_r_dimensionless(self, xy):

        mask_iron = self.iron_mask(xy)

        return (
            self.nu_air
            +
            (
                self.nu_iron
                - self.nu_air
            )
            * mask_iron
        )

    def mu_r_dimensionless(self, xy):

        return 1.0 / self.nu_r_dimensionless(xy)


# ============================================================
# Boundary condition
# ============================================================

class BoundaryCondition:

    def __init__(
        self,
        x_min=-DOMAIN_EXTENT,
        x_max=DOMAIN_EXTENT,
        y_min=-DOMAIN_EXTENT,
        y_max=DOMAIN_EXTENT,
        num_points=1000
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.num_points = num_points

    def sample_points(self, device=DEVICE):

        x_bottom = torch.empty(
            self.num_points,
            1,
            device=device,
            dtype=DTYPE
        ).uniform_(
            self.x_min,
            self.x_max
        )

        y_bottom = torch.full(
            (self.num_points, 1),
            self.y_min,
            device=device,
            dtype=DTYPE
        )

        x_top = torch.empty(
            self.num_points,
            1,
            device=device,
            dtype=DTYPE
        ).uniform_(
            self.x_min,
            self.x_max
        )

        y_top = torch.full(
            (self.num_points, 1),
            self.y_max,
            device=device,
            dtype=DTYPE
        )

        x_left = torch.full(
            (self.num_points, 1),
            self.x_min,
            device=device,
            dtype=DTYPE
        )

        y_left = torch.empty(
            self.num_points,
            1,
            device=device,
            dtype=DTYPE
        ).uniform_(
            self.y_min,
            self.y_max
        )

        x_right = torch.full(
            (self.num_points, 1),
            self.x_max,
            device=device,
            dtype=DTYPE
        )

        y_right = torch.empty(
            self.num_points,
            1,
            device=device,
            dtype=DTYPE
        ).uniform_(
            self.y_min,
            self.y_max
        )

        bottom = torch.cat(
            [x_bottom, y_bottom],
            dim=1
        )

        top = torch.cat(
            [x_top, y_top],
            dim=1
        )

        left = torch.cat(
            [x_left, y_left],
            dim=1
        )

        right = torch.cat(
            [x_right, y_right],
            dim=1
        )

        return torch.cat(
            [
                bottom,
                top,
                left,
                right
            ],
            dim=0
        )

    @staticmethod
    def max_error(
        model,
        points
    ):

        with torch.no_grad():
            Az = model(points)

        return torch.max(
            torch.abs(Az)
        ).item()


# ============================================================
# PINN model
# ============================================================

class MagneticPINN(nn.Module):

    def __init__(
        self,
        hidden_layers=6,
        neurons=128,
        hard_boundary=True
    ):
        super().__init__()

        self.hard_boundary = hard_boundary

        layers = [
            nn.Linear(2, neurons),
            nn.Tanh()
        ]

        for _ in range(
            hidden_layers - 1
        ):
            layers += [
                nn.Linear(
                    neurons,
                    neurons
                ),
                nn.Tanh()
            ]

        layers.append(
            nn.Linear(
                neurons,
                1
            )
        )

        self.net = nn.Sequential(
            *layers
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for module in self.net:

            if isinstance(
                module,
                nn.Linear
            ):

                nn.init.xavier_normal_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def boundary_factor(self, xy):

        x = xy[:, 0:1]
        y = xy[:, 1:2]

        return (
            (1.0 - (x / DOMAIN_EXTENT)**2)
            *
            (1.0 - (y / DOMAIN_EXTENT)**2)
        )

    def forward(self, xy):

        raw = self.net(xy)

        if self.hard_boundary:

            return (
                self.boundary_factor(xy)
                * raw
            )

        return raw


# ============================================================
# Sobol sampling
# ============================================================

def create_sobol_engine(seed=SEED):

    return torch.quasirandom.SobolEngine(
        dimension=2,
        scramble=True,
        seed=seed
    )


def sample_collocation_points(
    num_points,
    x_min=-DOMAIN_EXTENT,
    x_max=DOMAIN_EXTENT,
    y_min=-DOMAIN_EXTENT,
    y_max=DOMAIN_EXTENT,
    device=DEVICE,
    engine=None
):

    if engine is None:
        engine = create_sobol_engine()

    uv = engine.draw(
        num_points
    ).to(
        device=device,
        dtype=DTYPE
    )

    x = (
        x_min
        +
        (x_max - x_min)
        * uv[:, 0:1]
    )

    y = (
        y_min
        +
        (y_max - y_min)
        * uv[:, 1:2]
    )

    xy = torch.cat(
        [x, y],
        dim=1
    )

    xy.requires_grad_(True)

    return xy


# ============================================================
# PDE residual
# ============================================================

def compute_pde_loss(
    model,
    xy_collocation,
    source,
    material
):

    Az_star = model(
        xy_collocation
    )

    grads = torch.autograd.grad(
        outputs=Az_star,
        inputs=xy_collocation,
        grad_outputs=torch.ones_like(
            Az_star
        ),
        create_graph=True
    )[0]

    dAz_dx = grads[:, 0:1]
    dAz_dy = grads[:, 1:2]

    nu_r = material.nu_r_dimensionless(
        xy_collocation
    )

    flux_x = (
        nu_r
        * dAz_dx
    )

    flux_y = (
        nu_r
        * dAz_dy
    )

    div_flux_x = torch.autograd.grad(
        outputs=flux_x,
        inputs=xy_collocation,
        grad_outputs=torch.ones_like(
            flux_x
        ),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]

    div_flux_y = torch.autograd.grad(
        outputs=flux_y,
        inputs=xy_collocation,
        grad_outputs=torch.ones_like(
            flux_y
        ),
        create_graph=True
    )[0][:, 1:2]

    dMx_dy = source.dmx_dy_dimensionless(
        xy_collocation
    )

    residual = (
        div_flux_x
        +
        div_flux_y
        -
        dMx_dy
    )

    loss = torch.mean(
        residual**2
    )

    return loss, residual


# ============================================================
# Field evaluation
# ============================================================

def evaluate_fields(
    model,
    resolution=160,
    device=DEVICE
):

    was_training = model.training

    model.eval()

    x_star = np.linspace(
        -DOMAIN_EXTENT,
        DOMAIN_EXTENT,
        resolution,
        dtype=np.float32
    )

    y_star = np.linspace(
        -DOMAIN_EXTENT,
        DOMAIN_EXTENT,
        resolution,
        dtype=np.float32
    )

    X_star, Y_star = np.meshgrid(
        x_star,
        y_star
    )

    xy_np = np.column_stack(
        (
            X_star.ravel(),
            Y_star.ravel()
        )
    ).astype(
        np.float32
    )

    xy = torch.tensor(
        xy_np,
        device=device,
        dtype=DTYPE,
        requires_grad=True
    )

    Az_star = model(xy)

    grads = torch.autograd.grad(
        outputs=Az_star,
        inputs=xy,
        grad_outputs=torch.ones_like(
            Az_star
        ),
        create_graph=False
    )[0]

    dAz_dx = grads[:, 0:1]
    dAz_dy = grads[:, 1:2]

    Az = (
        A0 * Az_star
    ).detach().cpu().numpy().reshape(
        X_star.shape
    )

    Bx = (
        B0 * dAz_dy
    ).detach().cpu().numpy().reshape(
        X_star.shape
    )

    By = (
        -B0 * dAz_dx
    ).detach().cpu().numpy().reshape(
        X_star.shape
    )

    Bmag = np.sqrt(
        Bx**2
        +
        By**2
    )

    source = MagnetizationSource()
    material = MaterialPermeability()

    xy_detached = xy.detach()

    with torch.no_grad():

        Mx = (
            source.physical(
                xy_detached
            )
            .cpu()
            .numpy()
            .reshape(X_star.shape)
        )

        dMx_dy = (
            source.dmx_dy_physical(
                xy_detached
            )
            .cpu()
            .numpy()
            .reshape(X_star.shape)
        )

        Mur = (
            material.mu_r_dimensionless(
                xy_detached
            )
            .cpu()
            .numpy()
            .reshape(X_star.shape)
        )

    X = L0 * X_star
    Y = L0 * Y_star

    if was_training:
        model.train()

    return {
        "X": X,
        "Y": Y,
        "Az": Az,
        "Bx": Bx,
        "By": By,
        "Bmag": Bmag,
        "Mx": Mx,
        "dMx_dy": dMx_dy,
        "Mur": Mur
    }


# ============================================================
# div(B)
# ============================================================

def compute_divergence_B(
    model,
    xy
):

    Az_star = model(xy)

    grads = torch.autograd.grad(
        outputs=Az_star,
        inputs=xy,
        grad_outputs=torch.ones_like(
            Az_star
        ),
        create_graph=True
    )[0]

    dAz_dx = grads[:, 0:1]
    dAz_dy = grads[:, 1:2]

    Bx_star = dAz_dy
    By_star = -dAz_dx

    dBx_dx = torch.autograd.grad(
        outputs=Bx_star,
        inputs=xy,
        grad_outputs=torch.ones_like(
            Bx_star
        ),
        create_graph=False,
        retain_graph=True
    )[0][:, 0:1]

    dBy_dy = torch.autograd.grad(
        outputs=By_star,
        inputs=xy,
        grad_outputs=torch.ones_like(
            By_star
        ),
        create_graph=False
    )[0][:, 1:2]

    return (
        B0 / L0
    ) * (
        dBx_dx
        +
        dBy_dy
    )


# ============================================================
# Boundary flux
# ============================================================

def compute_boundary_flux(
    model,
    resolution=400,
    device=DEVICE
):

    model.eval()

    t = torch.linspace(
        -DOMAIN_EXTENT,
        DOMAIN_EXTENT,
        resolution,
        device=device,
        dtype=DTYPE
    ).reshape(-1, 1)

    y_bottom = torch.full_like(
        t,
        -DOMAIN_EXTENT
    )

    y_top = torch.full_like(
        t,
        DOMAIN_EXTENT
    )

    x_left = torch.full_like(
        t,
        -DOMAIN_EXTENT
    )

    x_right = torch.full_like(
        t,
        DOMAIN_EXTENT
    )

    bottom = torch.cat(
        [t, y_bottom],
        dim=1
    )

    top = torch.cat(
        [t, y_top],
        dim=1
    )

    left = torch.cat(
        [x_left, t],
        dim=1
    )

    right = torch.cat(
        [x_right, t],
        dim=1
    )

    def B_at(points):

        points = points.clone().requires_grad_(
            True
        )

        Az = model(points)

        grads = torch.autograd.grad(
            outputs=Az,
            inputs=points,
            grad_outputs=torch.ones_like(
                Az
            ),
            create_graph=False
        )[0]

        bx = (
            B0
            * grads[:, 1:2]
        )

        by = (
            -B0
            * grads[:, 0:1]
        )

        return (
            bx.detach()
            .cpu()
            .numpy()
            .ravel(),
            by.detach()
            .cpu()
            .numpy()
            .ravel()
        )

    bx_bottom, by_bottom = B_at(
        bottom
    )

    bx_top, by_top = B_at(
        top
    )

    bx_left, by_left = B_at(
        left
    )

    bx_right, by_right = B_at(
        right
    )

    t_np = (
        t.detach()
        .cpu()
        .numpy()
        .ravel()
    )

    phi_bottom = np.trapezoid(
        -by_bottom,
        t_np
    )

    phi_top = np.trapezoid(
        by_top,
        t_np
    )

    phi_left = np.trapezoid(
        -bx_left,
        t_np
    )

    phi_right = np.trapezoid(
        bx_right,
        t_np
    )

    total_flux = (
        phi_bottom
        +
        phi_top
        +
        phi_left
        +
        phi_right
    )

    return {
        "bottom": phi_bottom,
        "top": phi_top,
        "left": phi_left,
        "right": phi_right,
        "total": total_flux
    }


# ============================================================
# Diagnostics
# ============================================================

def report_diagnostics(
    model,
    source,
    material,
    num_validation=5000
):

    model.eval()

    bc = BoundaryCondition(
        num_points=1000
    )

    boundary_points = bc.sample_points(
        device=DEVICE
    )

    max_boundary_error = (
        bc.max_error(
            model,
            boundary_points
        )
    )

    validation_engine = create_sobol_engine(
        seed=SEED + 100
    )

    validation = sample_collocation_points(
        num_validation,
        device=DEVICE,
        engine=validation_engine
    )

    loss_pde, residual = compute_pde_loss(
        model,
        validation,
        source,
        material
    )

    residual_rms = torch.sqrt(
        torch.mean(
            residual**2
        )
    ).item()

    residual_max = torch.max(
        torch.abs(residual)
    ).item()

    div_B = compute_divergence_B(
        model,
        validation
    )

    div_B_rms = torch.sqrt(
        torch.mean(
            div_B**2
        )
    ).item()

    div_B_max = torch.max(
        torch.abs(div_B)
    ).item()

    fields = evaluate_fields(
        model,
        resolution=120,
        device=DEVICE
    )

    Bmag = fields["Bmag"]

    flux = compute_boundary_flux(
        model,
        resolution=300,
        device=DEVICE
    )

    print("\n========================================")
    print("             PINN DIAGNOSTICS")
    print("========================================")

    print(
        f"Device                 : {DEVICE}"
    )

    print(
        f"M0                     : "
        f"{M0:.6e} A/m"
    )

    print(
        f"L0                     : "
        f"{L0:.6e} m"
    )

    print(
        f"A0                     : "
        f"{A0:.6e} Wb/m"
    )

    print(
        f"B0                     : "
        f"{B0:.6e} T"
    )

    print(
        f"Iron mu_r              : "
        f"{MUR_IRON:.1f}"
    )

    print("----------------------------------------")

    print(
        f"Left polarity          : "
        f"{LEFT_POLARITY:+.1f}"
    )

    print(
        f"Right polarity         : "
        f"{RIGHT_POLARITY:+.1f}"
    )

    print("----------------------------------------")

    print(
        f"Boundary max |Az|      : "
        f"{max_boundary_error:.8e} Wb/m"
    )

    print(
        f"PDE MSE                : "
        f"{loss_pde.item():.8e}"
    )

    print(
        f"PDE RMS residual       : "
        f"{residual_rms:.8e}"
    )

    print(
        f"PDE max |residual|     : "
        f"{residual_max:.8e}"
    )

    print("----------------------------------------")

    print(
        f"div(B) RMS             : "
        f"{div_B_rms:.8e}"
    )

    print(
        f"div(B) max             : "
        f"{div_B_max:.8e}"
    )

    print(
        f"Bmax                   : "
        f"{np.max(Bmag):.8e} T"
    )

    print("----------------------------------------")

    print(
        f"Flux bottom            : "
        f"{flux['bottom']:.8e} Wb/m"
    )

    print(
        f"Flux top               : "
        f"{flux['top']:.8e} Wb/m"
    )

    print(
        f"Flux left              : "
        f"{flux['left']:.8e} Wb/m"
    )

    print(
        f"Flux right             : "
        f"{flux['right']:.8e} Wb/m"
    )

    print(
        f"Closed-boundary flux   : "
        f"{flux['total']:.8e} Wb/m"
    )

    print("========================================\n")


# ============================================================
# Training
# ============================================================

def train_pinn(
    epochs=4000,
    num_collocation=10000,
    hidden_layers=6,
    neurons=128,
    learning_rate=1.0e-3,
    print_every=250,
    lbfgs_collocation=5000,
    lbfgs_max_iter=200,
    lbfgs_max_eval=250,
    grad_clip=1.0
):

    model = MagneticPINN(
        hidden_layers=hidden_layers,
        neurons=neurons,
        hard_boundary=True
    ).to(DEVICE)

    source = MagnetizationSource()

    material = MaterialPermeability()

    adam = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            adam,
            T_max=epochs,
            eta_min=1.0e-6
        )
    )

    engine = create_sobol_engine(
        seed=SEED
    )

    history = {
        "adam_epoch": [],
        "adam_pde": [],
        "adam_lr": [],
        "lbfgs_step": [],
        "lbfgs_pde": []
    }

    model.train()

    print("\n========================================")
    print("                  ADAM")
    print("========================================")

    for epoch in range(
        1,
        epochs + 1
    ):

        adam.zero_grad(
            set_to_none=True
        )

        xy_collocation = (
            sample_collocation_points(
                num_collocation,
                device=DEVICE,
                engine=engine
            )
        )

        loss_pde, _ = compute_pde_loss(
            model,
            xy_collocation,
            source,
            material
        )

        loss_pde.backward()

        if grad_clip is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

        adam.step()

        scheduler.step()

        lr = adam.param_groups[0]["lr"]

        history["adam_epoch"].append(
            epoch
        )

        history["adam_pde"].append(
            loss_pde.item()
        )

        history["adam_lr"].append(
            lr
        )

        if (
            epoch == 1
            or epoch % print_every == 0
        ):

            print(
                f"Epoch {epoch:05d} | "
                f"PDE = "
                f"{loss_pde.item():.6e} | "
                f"LR = {lr:.3e}"
            )

    print("\n========================================")
    print("                 L-BFGS")
    print("========================================")

    lbfgs_count = min(
        lbfgs_collocation,
        num_collocation
    )

    lbfgs_engine = create_sobol_engine(
        seed=SEED + 1
    )

    xy_fixed = sample_collocation_points(
        lbfgs_count,
        device=DEVICE,
        engine=lbfgs_engine
    )

    lbfgs = optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=lbfgs_max_iter,
        max_eval=lbfgs_max_eval,
        history_size=50,
        tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10,
        line_search_fn="strong_wolfe"
    )

    lbfgs_counter = [0]

    def closure():

        lbfgs.zero_grad(
            set_to_none=True
        )

        loss_pde, _ = compute_pde_loss(
            model,
            xy_fixed,
            source,
            material
        )

        loss_pde.backward()

        lbfgs_counter[0] += 1

        history["lbfgs_step"].append(
            lbfgs_counter[0]
        )

        history["lbfgs_pde"].append(
            loss_pde.item()
        )

        if (
            lbfgs_counter[0] == 1
            or lbfgs_counter[0] % 20 == 0
        ):

            print(
                f"Step "
                f"{lbfgs_counter[0]:05d} | "
                f"PDE = "
                f"{loss_pde.item():.6e}"
            )

        return loss_pde

    lbfgs.step(
        closure
    )

    return model, history


# ============================================================
# Plot
# ============================================================

def plot_results(
    model,
    history,
    resolution=160
):

    fields = evaluate_fields(
        model,
        resolution=resolution,
        device=DEVICE
    )

    X = fields["X"]
    Y = fields["Y"]

    Az = fields["Az"]

    Bx = fields["Bx"]
    By = fields["By"]
    Bmag = fields["Bmag"]

    Mx = fields["Mx"]
    dMx_dy = fields["dMx_dy"]
    Mur = fields["Mur"]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(22, 18)
    )

    # --------------------------------------------------------
    # A_z
    # --------------------------------------------------------

    contour = axes[0, 0].contourf(
        X,
        Y,
        Az,
        levels=60,
        cmap="jet"
    )

    fig.colorbar(
        contour,
        ax=axes[0, 0],
        label="A_z (Wb/m)"
    )

    axes[0, 0].set_title(
        "Magnetic Vector Potential $A_z$"
    )

    # --------------------------------------------------------
    # |B|
    # --------------------------------------------------------

    contour = axes[0, 1].contourf(
        X,
        Y,
        Bmag,
        levels=60,
        cmap="jet"
    )

    fig.colorbar(
        contour,
        ax=axes[0, 1],
        label="|B| (T)"
    )

    axes[0, 1].set_title(
        "Magnetic Flux Density $|B|$"
    )

    # --------------------------------------------------------
    # Mx
    # --------------------------------------------------------

    contour = axes[0, 2].contourf(
        X,
        Y,
        Mx,
        levels=60,
        cmap="viridis"
    )

    fig.colorbar(
        contour,
        ax=axes[0, 2],
        label="$M_x$ (A/m)"
    )

    axes[0, 2].set_title(
        "Magnetization $M_x$"
    )

    # --------------------------------------------------------
    # B vector
    # --------------------------------------------------------

    step = max(
        1,
        resolution // 25
    )

    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]

    Bx_sub = Bx[::step, ::step]
    By_sub = By[::step, ::step]

    Bsub = np.sqrt(
        Bx_sub**2
        +
        By_sub**2
    )

    scale = np.maximum(
        Bsub,
        1.0e-12
    )

    Bx_dir = (
        Bx_sub / scale
    )

    By_dir = (
        By_sub / scale
    )

    axes[1, 0].contourf(
        X,
        Y,
        Bmag,
        levels=40,
        cmap="jet",
        alpha=0.25
    )

    axes[1, 0].quiver(
        X_sub,
        Y_sub,
        Bx_dir,
        By_dir,
        scale=25.0,
        pivot="mid"
    )

    axes[1, 0].set_title(
        "Magnetic Flux Density Vector $\\mathbf{B}$"
    )

    # --------------------------------------------------------
    # dMx/dy
    # --------------------------------------------------------

    contour = axes[1, 1].contourf(
        X,
        Y,
        dMx_dy,
        levels=60,
        cmap="coolwarm"
    )

    fig.colorbar(
        contour,
        ax=axes[1, 1],
        label=(
            r"$\partial M_x/\partial y$ "
            r"(A/m$^2$)"
        )
    )

    axes[1, 1].set_title(
        "Magnetization Source Term"
    )

    # --------------------------------------------------------
    # mu_r
    # --------------------------------------------------------

    contour = axes[1, 2].contourf(
        X,
        Y,
        Mur,
        levels=60,
        cmap="copper"
    )

    fig.colorbar(
        contour,
        ax=axes[1, 2],
        label=r"$\mu_r$"
    )

    axes[1, 2].set_title(
        "Relative Permeability $\\mu_r$"
    )

    # --------------------------------------------------------
    # Material map
    # --------------------------------------------------------

    material_map = np.zeros_like(
        X
    )

    mag_mask = (
        np.abs(Mx)
        >
        0.1 * M0
    )

    iron_mask = (
        Mur > 10.0
    )

    material_map[
        iron_mask
    ] = 2.0

    material_map[
        mag_mask & ~iron_mask
    ] = 1.0

    contour = axes[2, 0].contourf(
        X,
        Y,
        material_map,
        levels=[
            -0.5,
            0.5,
            1.5,
            2.5
        ],
        cmap="tab10"
    )

    cbar = fig.colorbar(
        contour,
        ax=axes[2, 0],
        ticks=[0, 1, 2]
    )

    cbar.ax.set_yticklabels(
        [
            "Air",
            "Magnets",
            "Iron Yoke"
        ]
    )

    axes[2, 0].set_title(
        "Material Map"
    )

    # --------------------------------------------------------
    # Magnetic charge density
    # --------------------------------------------------------

    dx = (
        X[0, 1]
        -
        X[0, 0]
    )

    rho_m = -np.gradient(
        Mx,
        dx,
        axis=1
    )

    contour = axes[2, 1].contourf(
        X,
        Y,
        rho_m,
        levels=60,
        cmap="bwr"
    )

    fig.colorbar(
        contour,
        ax=axes[2, 1],
        label=(
            r"$\rho_m=-\partial M_x/\partial x$ "
            r"(A/m$^2$)"
        )
    )

    axes[2, 1].set_title(
        "Magnetic Charge Density"
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    adam_pde = np.asarray(
        history["adam_pde"]
    )

    lbfgs_pde = np.asarray(
        history["lbfgs_pde"]
    )

    if adam_pde.size > 0:

        axes[2, 2].semilogy(
            np.arange(
                1,
                adam_pde.size + 1
            ),
            np.maximum(
                adam_pde,
                1.0e-20
            ),
            label="Adam"
        )

    if lbfgs_pde.size > 0:

        start = (
            adam_pde.size
            + 1
        )

        axes[2, 2].semilogy(
            np.arange(
                start,
                start + lbfgs_pde.size
            ),
            np.maximum(
                lbfgs_pde,
                1.0e-20
            ),
            label="L-BFGS"
        )

    axes[2, 2].set_title(
        "PDE Training History"
    )

    axes[2, 2].set_xlabel(
        "Training evaluation"
    )

    axes[2, 2].set_ylabel(
        "PDE loss"
    )

    axes[2, 2].legend()

    axes[2, 2].grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    plot_axes = [
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
        axes[1, 2],
        axes[2, 0],
        axes[2, 1]
    ]

    for ax in plot_axes:

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        ax.set_xlabel(
            "x (m)"
        )

        ax.set_ylabel(
            "y (m)"
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("          MAGNET CONFIGURATION")
    print("========================================")

    print(
        f"Left magnet polarity  : "
        f"{LEFT_POLARITY:+.1f}"
    )

    print(
        f"Right magnet polarity : "
        f"{RIGHT_POLARITY:+.1f}"
    )

    print("========================================")

    trained_model, history = train_pinn(
        epochs=4000,
        num_collocation=10000,
        hidden_layers=6,
        neurons=128,
        learning_rate=1.0e-3,
        print_every=250,
        lbfgs_collocation=5000,
        lbfgs_max_iter=200,
        lbfgs_max_eval=250,
        grad_clip=1.0
    )

    source = MagnetizationSource()
    material = MaterialPermeability()

    report_diagnostics(
        trained_model,
        source,
        material,
        num_validation=5000
    )

    plot_results(
        trained_model,
        history,
        resolution=160
    )