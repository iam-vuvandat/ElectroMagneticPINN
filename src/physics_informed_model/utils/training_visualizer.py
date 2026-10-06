import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio

class TrainingVisualizer:
    def __init__(self, parent_pinn):
        self.parent_pinn = parent_pinn
        self.output_directory = self.parent_pinn.visualization_config.output_directory
        self.x_bounds = self.parent_pinn.sampler_config.x_boundaries_tuple
        self.y_bounds = self.parent_pinn.sampler_config.y_boundaries_tuple
        self.resolution = self.parent_pinn.visualization_config.resolution
        
        self.frame_paths = []
        self.loss_history = []
        self.epoch_history = []
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        x_coords = np.linspace(self.x_bounds[0], self.x_bounds[1], self.resolution)
        y_coords = np.linspace(self.y_bounds[0], self.y_bounds[1], self.resolution)
        self.X_grid, self.Y_grid = np.meshgrid(x_coords, y_coords)
        self.eval_points_tensor = torch.tensor(
            np.column_stack((self.X_grid.ravel(), self.Y_grid.ravel())), 
            dtype=torch.float32
        )

    def save_frame(self, epoch, loss_value, training_points_tensor, nu_tensor, jz_tensor, hcx_tensor, hcy_tensor):
        self.loss_history.append(loss_value)
        self.epoch_history.append(epoch)
        
        model = self.parent_pinn.pinn_architecture_instance
        pde_evaluator = self.parent_pinn.maxwell_pde_loss_instance
        
        model.eval()
        computation_device = next(model.parameters()).device
        
        eval_points = self.eval_points_tensor.to(computation_device).clone().requires_grad_(True)
        
        # Mảng thông số vật lý tĩnh để tính toán PDE Residual
        nu_grid = torch.full((eval_points.shape[0], 1), pde_evaluator.nu0, device=computation_device)
        jz_grid = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        hcx_grid = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        hcy_grid = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        
        # Tính PDE Residual (dựa trên raw prediction A_z_star)
        A_z_star = model(eval_points)
        residual_pred = pde_evaluator.compute_residual(eval_points, A_z_star, nu_grid, jz_grid, hcx_grid, hcy_grid)
        
        # Tính các thông số Vật lý thực tế (A_z_phys, B_x, B_y) thông qua evaluate_fields
        A_z_phys, B_x_phys, B_y_phys = self.parent_pinn.evaluate_fields(eval_points)
        
        # Chuyển đổi sang Numpy để vẽ
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy**2 + B_y_numpy**2)
        
        residual_numpy = residual_pred.detach().cpu().numpy().reshape(self.resolution, self.resolution)
        points_numpy = training_points_tensor.detach().cpu().numpy()
        
        # Mở rộng layout thành 2 hàng x 3 cột (Kích thước 18x10)
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # [Hàng 1 - Cột 1]: Collocation Points
        axs[0, 0].scatter(points_numpy[:, 0], points_numpy[:, 1], s=1, c='black', alpha=0.5)
        axs[0, 0].set_title(f"Collocation Points")
        axs[0, 0].set_xlim(self.x_bounds)
        axs[0, 0].set_ylim(self.y_bounds)
        axs[0, 0].set_aspect('equal')
        
        # [Hàng 1 - Cột 2]: PDE Residual Error
        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, np.abs(residual_numpy), levels=50, cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=1e1))
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')
        
        # [Hàng 1 - Cột 3]: Loss Optimization
        axs[0, 2].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 2].set_yscale('log')
        axs[0, 2].set_title("Loss Optimization")
        axs[0, 2].set_xlabel("Epoch")
        axs[0, 2].set_ylabel("Loss")
        axs[0, 2].grid(True, which="both", ls="-", alpha=0.2)
        
        # [Hàng 2 - Cột 1]: Magnetic Vector Potential (Az)
        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=50, cmap="jet")
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        # [Hàng 2 - Cột 2]: Flux Density Magnitude (|B|)
        contour_b = axs[1, 1].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=50, cmap="rainbow")
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title("Flux Density Magnitude ($|B|$)")
        axs[1, 1].set_aspect('equal')

        # [Hàng 2 - Cột 3]: Flux Density Vectors (Quiver Plot)
        contour_b_bg = axs[1, 2].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=50, cmap="rainbow", alpha=0.4)
        fig.colorbar(contour_b_bg, ax=axs[1, 2], label="|B| (T)")
        
        # Tính toán mật độ mũi tên sao cho không bị rối mắt
        step = max(1, self.resolution // 20)
        axs[1, 2].quiver(self.X_grid[::step, ::step], self.Y_grid[::step, ::step], 
                         B_x_numpy[::step, ::step], B_y_numpy[::step, ::step], 
                         color='black', pivot='mid')
        axs[1, 2].set_title("Flux Density Vectors (B)")
        axs[1, 2].set_aspect('equal')
        
        fig.suptitle(f"PINN Training Process - Epoch {epoch}", fontsize=16)
        plt.tight_layout()
        
        frame_name = os.path.join(self.output_directory, f"frame_{epoch:05d}.png")
        plt.savefig(frame_name, dpi=100)
        plt.close(fig)
        
        # Đưa model quay lại chế độ train
        model.train()

    def generate_gif(self):
        output_filename = self.parent_pinn.visualization_config.gif_filename
        fps = self.parent_pinn.visualization_config.gif_fps
        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")