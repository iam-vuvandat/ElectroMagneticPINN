import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio

class TrainingVisualizer:
    def __init__(self, electro_magnetic_pinn=None, parent_pinn=None):
        self.electro_magnetic_pinn = electro_magnetic_pinn if electro_magnetic_pinn is not None else parent_pinn
        self.parent_pinn = self.electro_magnetic_pinn
        
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        vis_cfg = self.electro_magnetic_pinn.visualization_config
        
        self.output_directory = vis_cfg.output_directory
        self.x_bounds = sim_domain.x_boundaries_tuple
        self.y_bounds = sim_domain.y_boundaries_tuple
        self.resolution = vis_cfg.resolution
        
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

    def save_frame(self, epoch, loss_value, domains_data):
        self.loss_history.append(loss_value)
        self.epoch_history.append(epoch)
        
        model = self.electro_magnetic_pinn.pinn_architecture_instance
        pde_evaluator = self.electro_magnetic_pinn.maxwell_pde_loss_instance
        sim_domain = self.electro_magnetic_pinn.simulation_domain_engine_instance
        
        model.eval()
        computation_device = next(model.parameters()).device
        eval_points = self.eval_points_tensor.to(computation_device).detach().clone().requires_grad_(True)
        
        phys_props = sim_domain.evaluate_global_physical_properties(eval_points)
        mat_id = phys_props["material_classification"]
        
        A_z_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_x_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        B_y_phys = torch.zeros((eval_points.shape[0], 1), device=computation_device)
        residual_total = torch.zeros((eval_points.shape[0], 1), device=computation_device)

        for domain in sim_domain.domains_list:
            d_id = domain.domain_id
            net_key = 'background' if getattr(domain, 'domain_name', '') == 'background' else f'domain_{d_id}'
            mask_bool = (mat_id == float(d_id))
            if mask_bool.any():
                Az_sub_star = model.forward_network(net_key, eval_points)
                grad_sub_star = torch.autograd.grad(
                    outputs=Az_sub_star,
                    inputs=eval_points,
                    grad_outputs=torch.ones_like(Az_sub_star),
                    create_graph=True
                )[0]
                Az_sub_phys = Az_sub_star * sim_domain.A0
                Bx_sub_phys = grad_sub_star[:, 1:2] * sim_domain.A0
                By_sub_phys = -grad_sub_star[:, 0:1] * sim_domain.A0

                nu_dom = domain.evaluate_reluctivity(eval_points).to(computation_device)
                hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
                hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
                jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

                res_sub = pde_evaluator.compute_residual(
                    eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom
                )

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
                residual_total = torch.where(mask_bool, torch.abs(res_sub.detach()), residual_total)
        
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
        residual_numpy = residual_total.cpu().numpy().reshape(self.resolution, self.resolution)
        
        all_pts = []
        for key, d in domains_data.items():
            if d["interior_points"].shape[0] > 0:
                all_pts.append(d["interior_points"].detach().cpu().numpy())
            if d["surface_points"].shape[0] > 0:
                all_pts.append(d["surface_points"].detach().cpu().numpy())
        points_numpy = np.vstack(all_pts) if all_pts else np.empty((0, 2))
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        if points_numpy.shape[0] > 0:
            axs[0, 0].scatter(points_numpy[:, 0], points_numpy[:, 1], s=1, c='black', alpha=0.5)
        axs[0, 0].set_title("Collocation Points")
        axs[0, 0].set_xlim(self.x_bounds)
        axs[0, 0].set_ylim(self.y_bounds)
        axs[0, 0].set_aspect('equal')
        
        res_abs = np.clip(np.abs(residual_numpy), 1e-6, 1e2)
        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, res_abs, levels=50, cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=1e1))
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')
        
        axs[0, 2].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 2].set_yscale('log')
        axs[0, 2].set_title("Loss Optimization")
        axs[0, 2].set_xlabel("Step / Epoch")
        axs[0, 2].set_ylabel("Loss")
        axs[0, 2].grid(True, which="both", ls="-", alpha=0.2)
        
        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=50, cmap="jet")
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        contour_b = axs[1, 1].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=50, cmap="jet")
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title("Flux Density Magnitude ($|B|$)")
        axs[1, 1].set_aspect('equal')

        contour_b_bg = axs[1, 2].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=50, cmap="jet", alpha=0.3)
        fig.colorbar(contour_b_bg, ax=axs[1, 2], label="|B| (T)")
        
        step = max(1, self.resolution // 20)
        B_sub = np.maximum(B_mag_numpy[::step, ::step], 1e-12)
        axs[1, 2].quiver(
            self.X_grid[::step, ::step], self.Y_grid[::step, ::step], 
            B_x_numpy[::step, ::step] / B_sub, B_y_numpy[::step, ::step] / B_sub, 
            scale=30.0, color='black', pivot='mid'
        )
        axs[1, 2].set_title("Flux Density Vectors (B)")
        axs[1, 2].set_aspect('equal')
        
        fig.suptitle(f"PINN Training Process - Step {epoch}", fontsize=16)
        plt.tight_layout()
        
        frame_name = os.path.join(self.output_directory, f"frame_{len(self.frame_paths) + 1:05d}.png")
        plt.savefig(frame_name, dpi=100)
        plt.close(fig)
        
        self.frame_paths.append(frame_name)
        model.train()

    def generate_gif(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'gif_filename', 'training_process.gif')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo GIF.")
            return
            
        full_gif_path = os.path.join(self.output_directory, output_filename)
        images = []
        for filename in self.frame_paths:
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
                
        imageio.mimsave(full_gif_path, images, fps=fps, loop=0)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")

    def generate_mp4(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        codec = getattr(cfg, 'mp4_codec', 'libx264')
        quality = getattr(cfg, 'mp4_quality', 8)

        if not self.frame_paths:
            print("[!] CẢNH BÁO: Không có khung hình tĩnh nào để tạo MP4.")
            return

        valid_frames = [f for f in self.frame_paths if os.path.exists(f)]
        if not valid_frames:
            print("[!] CẢNH BÁO: Tất cả frame paths đều không tồn tại trên đĩa.")
            return

        full_mp4_path = os.path.join(self.output_directory, output_filename)
        print(f"[*] Đang xuất Video MP4 ({len(valid_frames)} frames, {fps} fps)...")
        
        exported = False
        try:
            writer = imageio.get_writer(
                full_mp4_path,
                fps=fps,
                codec=codec,
                pixelformat='yuv420p',
                quality=quality
            )
            for f in valid_frames:
                writer.append_data(imageio.imread(f))
            writer.close()
            exported = True
            print(f"[*] Đã xuất MP4 Video thành công (imageio) tại: {full_mp4_path}")
        except Exception as exc:
            print(f"[!] Không thể xuất MP4 bằng imageio-ffmpeg ({exc}). Chuyển sang OpenCV fallback...")

        if not exported:
            try:
                import cv2
                sample = cv2.imread(valid_frames[0])
                height, width, _ = sample.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(full_mp4_path, fourcc, float(fps), (width, height))
                for f in valid_frames:
                    img = cv2.imread(f)
                    video_writer.write(img)
                video_writer.release()
                print(f"[*] Đã xuất MP4 Video thành công (OpenCV) tại: {full_mp4_path}")
            except Exception as cv_exc:
                print(f"[!] Lỗi khi xuất video MP4 bằng OpenCV: {cv_exc}")

    def generate_animations(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        gen_gif = getattr(cfg, 'generate_gif_animation', True)
        gen_mp4 = getattr(cfg, 'generate_mp4_video', True)

        if gen_gif:
            self.generate_gif()
        if gen_mp4:
            self.generate_mp4()
