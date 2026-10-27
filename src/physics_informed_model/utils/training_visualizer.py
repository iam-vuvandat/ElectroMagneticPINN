import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio
from simulation_domain_engine.global_physical_properties_evaluation import VACUUM_RELUCTIVITY

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
        nu_phys_all = torch.zeros((eval_points.shape[0], 1), device=computation_device)
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

                B_mag_sub_phys = torch.sqrt(Bx_sub_phys**2 + By_sub_phys**2 + 1e-12)

                nu_dom = domain.evaluate_reluctivity(eval_points, B_magnitude_tensor=B_mag_sub_phys).to(computation_device)
                hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
                hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
                jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

                res_lin, res_nonlin = pde_evaluator.compute_residual_components(
                    eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom, domain.evaluate_reluctivity(eval_points, is_nonlinear=False).to(computation_device)
                )

                A_z_phys = torch.where(mask_bool, Az_sub_phys.detach(), A_z_phys)
                B_x_phys = torch.where(mask_bool, Bx_sub_phys.detach(), B_x_phys)
                B_y_phys = torch.where(mask_bool, By_sub_phys.detach(), B_y_phys)
                nu_phys_all = torch.where(mask_bool, nu_dom.detach(), nu_phys_all)
                residual_total = torch.where(mask_bool, torch.abs((res_lin + res_nonlin).detach()), residual_total)
        
        A_z_numpy = A_z_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_x_numpy = B_x_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_y_numpy = B_y_phys.cpu().numpy().reshape(self.resolution, self.resolution)
        B_mag_numpy = np.sqrt(B_x_numpy ** 2 + B_y_numpy ** 2)
        mu_r_numpy = (VACUUM_RELUCTIVITY / torch.clamp(nu_phys_all, min=1e-6)).cpu().numpy().reshape(self.resolution, self.resolution)
        residual_numpy = residual_total.cpu().numpy().reshape(self.resolution, self.resolution)
        
        # Grid layout 2x4
        fig, axs = plt.subplots(2, 4, figsize=(23, 10))
        
        # =========================================================================
        # [0, 0] Categorized Collocation Points (Phân loại chi tiết từng tập điểm)
        # =========================================================================
        ax_pts = axs[0, 0]
        ax_pts.set_title("Categorized Collocation Points")
        
        domain_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
        pair_colors = ['#d62728', '#e377c2', '#bcbd22', '#17becf', '#ff1493']

        if "domains" in domains_data:
            domains_dict = domains_data["domains"]
            shared_dict = domains_data.get("shared_interfaces", {})
        else:
            domains_dict = domains_data
            shared_dict = {}

        # 1. Tô màu điểm nội miền và điểm biên ngoài của từng miền
        for idx, (key, d) in enumerate(domains_dict.items()):
            dom_obj = d["domain_object"]
            d_name = getattr(dom_obj, "domain_name", key)
            c_dom = domain_colors[idx % len(domain_colors)]
            
            # Điểm nội miền
            int_pts = d.get("interior_points")
            if int_pts is not None and int_pts.shape[0] > 0:
                pts_np = int_pts.detach().cpu().numpy()
                ax_pts.scatter(pts_np[:, 0], pts_np[:, 1], s=2.5, color=c_dom, alpha=0.35, label=f"Int: {d_name}")

            # Điểm biên ngoài (tiếp xúc với không khí)
            outer_pts = d.get("outer_surface_points")
            if outer_pts is not None and outer_pts.shape[0] > 0:
                opts_np = outer_pts.detach().cpu().numpy()
                ax_pts.scatter(opts_np[:, 0], opts_np[:, 1], s=8, color='gray', alpha=0.6, label=f"Air Boundary" if idx==0 else "")

        # 2. Tô màu nổi bật riêng biệt cho từng Cặp ranh giới tiếp xúc giữa các vật liệu khác nhau
        for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
            if_pts = pair_data.get("interface_points")
            if if_pts is not None and if_pts.shape[0] > 0:
                pts_np = if_pts.detach().cpu().numpy()
                c_pair = pair_colors[p_idx % len(pair_colors)]
                name_a = getattr(pair_data["domain_a"], "domain_name", pair_data["key_a"])
                name_b = getattr(pair_data["domain_b"], "domain_name", pair_data["key_b"])
                
                ax_pts.scatter(
                    pts_np[:, 0], pts_np[:, 1], 
                    s=20, color=c_pair, marker='x', linewidths=1.5, 
                    label=f"Interface: {name_a} ↔ {name_b}"
                )

        ax_pts.legend(loc='upper right', fontsize=6.5, framealpha=0.85)
        ax_pts.set_xlim(self.x_bounds)
        ax_pts.set_ylim(self.y_bounds)
        ax_pts.set_aspect('equal')
        
        # [0, 1] PDE Residual Error
        res_abs = np.clip(np.abs(residual_numpy), 1e-6, 1e2)
        contour_res = axs[0, 1].contourf(self.X_grid, self.Y_grid, res_abs, levels=50, cmap="Reds", norm=LogNorm(vmin=1e-4, vmax=1e1))
        fig.colorbar(contour_res, ax=axs[0, 1])
        axs[0, 1].set_title("PDE Residual Error")
        axs[0, 1].set_aspect('equal')

        # [0, 2] Dynamic Relative Permeability mu_r(B)
        contour_mur = axs[0, 2].contourf(self.X_grid, self.Y_grid, mu_r_numpy, levels=50, cmap="cividis")
        fig.colorbar(contour_mur, ax=axs[0, 2], label=r"Relative Permeability ($\mu_r$)")
        axs[0, 2].set_title(r"Permeability Distribution ($\mu_r$)")
        axs[0, 2].set_aspect('equal')
        
        # [0, 3] Loss Optimization
        axs[0, 3].plot(self.epoch_history, self.loss_history, 'b-')
        axs[0, 3].set_yscale('log')
        axs[0, 3].set_title("Loss Optimization")
        axs[0, 3].set_xlabel("Step / Epoch")
        axs[0, 3].set_ylabel("Loss")
        axs[0, 3].grid(True, which="both", ls="-", alpha=0.2)
        
        # [1, 0] Az
        contour_az = axs[1, 0].contourf(self.X_grid, self.Y_grid, A_z_numpy, levels=50, cmap="jet")
        fig.colorbar(contour_az, ax=axs[1, 0], label="A_z (Wb/m)")
        axs[1, 0].set_title("Magnetic Vector Potential ($A_z$)")
        axs[1, 0].set_aspect('equal')

        # [1, 1] |B|
        b_max_val = max(float(np.max(B_mag_numpy)), 1e-3)
        levels_b = np.linspace(0.0, b_max_val, 50)
        contour_b = axs[1, 1].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=levels_b, cmap="jet", vmin=0.0, vmax=b_max_val)
        fig.colorbar(contour_b, ax=axs[1, 1], label="|B| (T)")
        axs[1, 1].set_title("Flux Density Magnitude ($|B|$)")
        axs[1, 1].set_aspect('equal')

        # [1, 2] Flux Density Vectors
        contour_b_bg = axs[1, 2].contourf(self.X_grid, self.Y_grid, B_mag_numpy, levels=levels_b, cmap="jet", alpha=0.3, vmin=0.0, vmax=b_max_val)
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

        # [1, 3] Material Domain Layout
        mat_id_numpy = mat_id.cpu().numpy().reshape(self.resolution, self.resolution)
        im_mat = axs[1, 3].imshow(mat_id_numpy, extent=[self.x_bounds[0], self.x_bounds[1], self.y_bounds[0], self.y_bounds[1]], origin='lower', interpolation='nearest', cmap='tab10')
        fig.colorbar(im_mat, ax=axs[1, 3], label="Domain ID")
        axs[1, 3].set_title("Material Classification")
        axs[1, 3].set_aspect('equal')
        
        fig.suptitle(f"XPINN Training Process - Step {epoch}", fontsize=16)
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
        images = [imageio.imread(f) for f in self.frame_paths if os.path.exists(f)]
        imageio.mimsave(full_gif_path, images, fps=fps, loop=0)
        print(f"[*] Đã xuất GIF Animation thành công tại: {full_gif_path}")

    def generate_mp4(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        output_filename = getattr(cfg, 'mp4_filename', 'training_process.mp4')
        fps = getattr(cfg, 'animation_fps', getattr(cfg, 'gif_fps', 5))
        codec = getattr(cfg, 'mp4_codec', 'libx264')
        quality = getattr(cfg, 'mp4_quality', 8)

        valid_frames = [f for f in self.frame_paths if os.path.exists(f)]
        if not valid_frames:
            return

        full_mp4_path = os.path.join(self.output_directory, output_filename)
        try:
            writer = imageio.get_writer(full_mp4_path, fps=fps, codec=codec, pixelformat='yuv420p', quality=quality)
            for f in valid_frames:
                writer.append_data(imageio.imread(f))
            writer.close()
            print(f"[*] Đã xuất MP4 Video thành công tại: {full_mp4_path}")
        except Exception as exc:
            print(f"[!] Lỗi xuất MP4: {exc}")

    def generate_animations(self):
        cfg = self.electro_magnetic_pinn.visualization_config
        if getattr(cfg, 'generate_gif_animation', True):
            self.generate_gif()
        if getattr(cfg, 'generate_mp4_video', True):
            self.generate_mp4()