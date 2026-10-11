import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class SubNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, xy):
        return self.net(xy)

def compute_gradients(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

def main():
    torch.manual_seed(42)
    
    net_air = SubNet()
    net_mag = SubNet()
    
    optimizer = optim.Adam(list(net_air.parameters()) + list(net_mag.parameters()), lr=1e-3)
    
    nu_air = 1.0
    nu_mag = 0.1
    H_cy = 2.0
    
    points_mag = (torch.rand(1500, 2) * torch.tensor([1.0, 2.0]) - torch.tensor([0.5, 1.0])).requires_grad_(True)
    
    points_air_left = (torch.rand(1000, 2) * torch.tensor([0.5, 2.0]) - torch.tensor([1.0, 1.0])).requires_grad_(True)
    points_air_right = (torch.rand(1000, 2) * torch.tensor([0.5, 2.0]) - torch.tensor([-0.5, 1.0])).requires_grad_(True)
    points_air = torch.cat([points_air_left, points_air_right], dim=0)
    
    interface_left = torch.cat([torch.full((400, 1), -0.5), torch.rand(400, 1) * 2.0 - 1.0], dim=1).requires_grad_(True)
    interface_right = torch.cat([torch.full((400, 1), 0.5), torch.rand(400, 1) * 2.0 - 1.0], dim=1).requires_grad_(True)
    
    bc_top = torch.cat([torch.rand(400, 1) * 2.0 - 1.0, torch.full((400, 1), 1.0)], dim=1)
    bc_bottom = torch.cat([torch.rand(400, 1) * 2.0 - 1.0, torch.full((400, 1), -1.0)], dim=1)
    bc_left = torch.cat([torch.full((400, 1), -1.0), torch.rand(400, 1) * 2.0 - 1.0], dim=1)
    bc_right = torch.cat([torch.full((400, 1), 1.0), torch.rand(400, 1) * 2.0 - 1.0], dim=1)
    
    epochs = 4000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        Az_mag = net_mag(points_mag)
        grad_Az_mag = compute_gradients(Az_mag, points_mag)
        dAz_dx_mag = grad_Az_mag[:, 0:1]
        dAz_dy_mag = grad_Az_mag[:, 1:2]
        Hx_mag = nu_mag * dAz_dy_mag
        Hy_mag = -nu_mag * dAz_dx_mag - H_cy
        dHx_dy_mag = compute_gradients(Hx_mag, points_mag)[:, 1:2]
        dHy_dx_mag = compute_gradients(Hy_mag, points_mag)[:, 0:1]
        loss_pde_mag = torch.mean((dHy_dx_mag - dHx_dy_mag)**2)
        
        Az_air = net_air(points_air)
        grad_Az_air = compute_gradients(Az_air, points_air)
        dAz_dx_air = grad_Az_air[:, 0:1]
        dAz_dy_air = grad_Az_air[:, 1:2]
        Hx_air = nu_air * dAz_dy_air
        Hy_air = -nu_air * dAz_dx_air
        dHx_dy_air = compute_gradients(Hx_air, points_air)[:, 1:2]
        dHy_dx_air = compute_gradients(Hy_air, points_air)[:, 0:1]
        loss_pde_air = torch.mean((dHy_dx_air - dHx_dy_air)**2)
        
        Az_bc_top = torch.where((bc_top[:, 0:1] >= -0.5) & (bc_top[:, 0:1] <= 0.5), net_mag(bc_top), net_air(bc_top))
        Az_bc_bottom = torch.where((bc_bottom[:, 0:1] >= -0.5) & (bc_bottom[:, 0:1] <= 0.5), net_mag(bc_bottom), net_air(bc_bottom))
        Az_bc_left = net_air(bc_left)
        Az_bc_right = net_air(bc_right)
        loss_bc = torch.mean(Az_bc_top**2) + torch.mean(Az_bc_bottom**2) + torch.mean(Az_bc_left**2) + torch.mean(Az_bc_right**2)
        
        Az_I_L_mag = net_mag(interface_left)
        Az_I_L_air = net_air(interface_left)
        loss_I_val_L = torch.mean((Az_I_L_mag - Az_I_L_air)**2)
        
        Az_I_R_mag = net_mag(interface_right)
        Az_I_R_air = net_air(interface_right)
        loss_I_val_R = torch.mean((Az_I_R_mag - Az_I_R_air)**2)
        
        grad_I_L_mag = compute_gradients(Az_I_L_mag, interface_left)
        grad_I_L_air = compute_gradients(Az_I_L_air, interface_left)
        Hy_I_L_mag = -nu_mag * grad_I_L_mag[:, 0:1] - H_cy
        Hy_I_L_air = -nu_air * grad_I_L_air[:, 0:1]
        loss_I_flux_L = torch.mean((Hy_I_L_mag - Hy_I_L_air)**2)
        
        grad_I_R_mag = compute_gradients(Az_I_R_mag, interface_right)
        grad_I_R_air = compute_gradients(Az_I_R_air, interface_right)
        Hy_I_R_mag = -nu_mag * grad_I_R_mag[:, 0:1] - H_cy
        Hy_I_R_air = -nu_air * grad_I_R_air[:, 0:1]
        loss_I_flux_R = torch.mean((Hy_I_R_mag - Hy_I_R_air)**2)
        
        loss = loss_pde_mag + loss_pde_air + loss_bc + loss_I_val_L + loss_I_val_R + loss_I_flux_L + loss_I_flux_R
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6e}")

    x = np.linspace(-1, 1, 150)
    y = np.linspace(-1, 1, 150)
    X, Y = np.meshgrid(x, y)
    xy_tensor = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)
    
    with torch.no_grad():
        Az_mag_pred = net_mag(xy_tensor)
        Az_air_pred = net_air(xy_tensor)
        mask_mag = (xy_tensor[:, 0:1] >= -0.5) & (xy_tensor[:, 0:1] <= 0.5)
        Az_pred = torch.where(mask_mag, Az_mag_pred, Az_air_pred)
        Az_grid = Az_pred.numpy().reshape(150, 150)
        
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Az_grid, levels=60, cmap="jet")
    plt.colorbar(contour, label="A_z")
    plt.axvline(-0.5, color='white', linestyle='--', alpha=0.8)
    plt.axvline(0.5, color='white', linestyle='--', alpha=0.8)
    plt.title('2D Domain Decomposition (Magnet in Air)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('xpinn_2d_demo.png')
    print("[*] Hoàn thành. Kết quả biểu đồ được lưu tại 'xpinn_2d_demo.png'")

if __name__ == '__main__':
    main()