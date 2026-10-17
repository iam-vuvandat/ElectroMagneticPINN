import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class FieldNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def evaluate_reluctivity(B, b_sat, nu_0=800000.0, mu_r=1000.0):
    nu_min = nu_0 / mu_r
    B2 = B ** 2
    tau = b_sat ** 2
    return nu_min + (nu_0 - nu_min) * (B2 / (B2 + tau + 1e-8))

def compute_pde_residual(model, x, b_sat, J0=5e6):
    A = model(x)
    grad_A = torch.autograd.grad(
        outputs=A,
        inputs=x,
        grad_outputs=torch.ones_like(A),
        create_graph=True
    )[0]
    B = torch.abs(grad_A)
    nu = evaluate_reluctivity(B, b_sat)
    H = nu * grad_A
    grad_H = torch.autograd.grad(
        outputs=H,
        inputs=x,
        grad_outputs=torch.ones_like(H),
        create_graph=True
    )[0]
    residual = grad_H + J0
    return residual, A, B

def run_forward_problem():
    true_b_sat = torch.tensor(1.6)
    model_forward = FieldNetwork()
    optimizer = torch.optim.Adam(model_forward.parameters(), lr=2e-3)
    
    x_interior = torch.linspace(0.0, 0.1, 100).view(-1, 1).requires_grad_(True)
    x_bc = torch.tensor([[0.0], [0.1]], requires_grad=False)
    
    for epoch in range(2500):
        optimizer.zero_grad()
        res, _, _ = compute_pde_residual(model_forward, x_interior, true_b_sat)
        loss_pde = torch.mean(res ** 2) / (800.0 ** 2)
        
        A_bc = model_forward(x_bc)
        loss_bc = torch.mean(A_bc ** 2) * 1e4
        
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()
        
    return model_forward, true_b_sat

def run_inverse_problem(sensor_x, sensor_B_meas):
    model_inverse = FieldNetwork()
    raw_param = nn.Parameter(torch.tensor(0.0))
    
    def get_constrained_b_sat(raw_val):
        return 0.5 + (2.5 - 0.5) * torch.sigmoid(raw_val)

    optimizer = torch.optim.Adam([
        {'params': model_inverse.parameters(), 'lr': 2e-3},
        {'params': [raw_param], 'lr': 1e-2}
    ])
    
    x_interior = torch.linspace(0.0, 0.1, 100).view(-1, 1).requires_grad_(True)
    x_bc = torch.tensor([[0.0], [0.1]], requires_grad=False)
    
    b_sat_history = []
    
    for epoch in range(3500):
        optimizer.zero_grad()
        current_b_sat = get_constrained_b_sat(raw_param)
        
        res, _, _ = compute_pde_residual(model_inverse, x_interior, current_b_sat)
        loss_pde = torch.mean(res ** 2) / (800.0 ** 2)
        
        A_bc = model_inverse(x_bc)
        loss_bc = torch.mean(A_bc ** 2) * 1e4
        
        sensor_x_grad = sensor_x.clone().requires_grad_(True)
        A_sensor = model_inverse(sensor_x_grad)
        grad_A_sensor = torch.autograd.grad(
            outputs=A_sensor,
            inputs=sensor_x_grad,
            grad_outputs=torch.ones_like(A_sensor),
            create_graph=True
        )[0]
        B_sensor_pred = torch.abs(grad_A_sensor)
        loss_data = torch.mean((B_sensor_pred - sensor_B_meas) ** 2)
        
        loss = loss_pde + loss_bc + 50.0 * loss_data
        loss.backward()
        optimizer.step()
        
        b_sat_history.append(current_b_sat.item())
        
    return model_inverse, b_sat_history, current_b_sat.item()

def main():
    model_forward, true_b_sat = run_forward_problem()
    
    sensor_x = torch.tensor([[0.015], [0.035], [0.050], [0.065], [0.085]], requires_grad=True)
    A_sens = model_forward(sensor_x)
    grad_A_sens = torch.autograd.grad(
        outputs=A_sens,
        inputs=sensor_x,
        grad_outputs=torch.ones_like(A_sens)
    )[0]
    sensor_B_meas = torch.abs(grad_A_sens).detach()
    
    model_inverse, b_sat_history, recovered_b_sat = run_inverse_problem(sensor_x.detach(), sensor_B_meas)
    
    x_eval = torch.linspace(0.0, 0.1, 200).view(-1, 1).requires_grad_(True)
    
    A_fwd = model_forward(x_eval)
    B_fwd = torch.abs(torch.autograd.grad(A_fwd, x_eval, torch.ones_like(A_fwd), retain_graph=True)[0]).detach().numpy()
    A_fwd = A_fwd.detach().numpy()
    
    A_inv = model_inverse(x_eval)
    B_inv = torch.abs(torch.autograd.grad(A_inv, x_eval, torch.ones_like(A_inv), retain_graph=True)[0]).detach().numpy()
    A_inv = A_inv.detach().numpy()
    
    x_numpy = x_eval.detach().numpy().flatten()
    
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_numpy, A_fwd, label="Forward (True)")
    plt.plot(x_numpy, A_inv, '--', label="Inverse Recovered")
    plt.title("Magnetic Vector Potential A(x)")
    plt.xlabel("Position x (m)")
    plt.ylabel("A (Wb/m)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(x_numpy, B_fwd, label="Forward (True)")
    plt.plot(x_numpy, B_inv, '--', label="Inverse Recovered")
    plt.scatter(sensor_x.detach().numpy(), sensor_B_meas.numpy(), color='red', label="Sensors", zorder=5)
    plt.title("Flux Density B(x)")
    plt.xlabel("Position x (m)")
    plt.ylabel("|B| (Tesla)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(b_sat_history, color='purple', label="Estimated B_sat")
    plt.axhline(y=true_b_sat.item(), color='green', linestyle=':', label=f"True Value ({true_b_sat.item():.2f} T)")
    plt.title(f"Parameter Convergence (Final: {recovered_b_sat:.3f} T)")
    plt.xlabel("Epochs")
    plt.ylabel("B_sat (Tesla)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()