import os
import re
import sys

def get_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "..", "..")),
        "/content/ElectroMagneticPINN/src/physics_informed_model",
        "/content/ElectroMagneticPINN/physics_informed_model",
        os.getcwd()
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "training_engine")):
            return cand
    return os.path.abspath(os.path.join(script_dir, ".."))

def update_maxwell_pde_loss(base_dir):
    """1. Chuẩn hóa residual điểm theo nu_star cục bộ trong maxwell_pde_loss.py."""
    file_path = os.path.join(base_dir, "physics_domain", "physical_equations", "maxwell_pde_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_fn = """    def compute_linear_pde_residual(self, xy, A_z_star, nu_phys, J_z, H_cx, H_cy, nu_star_net=None):
        if nu_star_net is not None:
            nu_star = nu_star_net
        else:
            nu_star = nu_phys / self.nu0

        H_cx_star = H_cx / self.H0
        H_cy_star = H_cy / self.H0
        J_z_star = J_z / (self.H0 / self.L0)

        # 1. Gradient của Az* -> B*
        grad_A_star = torch.autograd.grad(
            outputs=A_z_star,
            inputs=xy,
            grad_outputs=torch.ones_like(A_z_star),
            create_graph=True
        )[0]

        dAz_dx_star = grad_A_star[:, 0:1] * self.L0
        dAz_dy_star = grad_A_star[:, 1:2] * self.L0

        # 2. H* = nu* * B* - Hc*
        Hx_star = nu_star * dAz_dy_star - H_cx_star
        Hy_star = -nu_star * dAz_dx_star - H_cy_star

        # 3. curl(H*) = dHy*/dx* - dHx*/dy*
        grad_Hx = torch.autograd.grad(outputs=Hx_star, inputs=xy, grad_outputs=torch.ones_like(Hx_star), create_graph=True)[0]
        grad_Hy = torch.autograd.grad(outputs=Hy_star, inputs=xy, grad_outputs=torch.ones_like(Hy_star), create_graph=True)[0]

        dHy_dx_star = grad_Hy[:, 0:1] * self.L0
        dHx_dy_star = grad_Hx[:, 1:2] * self.L0

        res_pde = dHy_dx_star - dHx_dy_star - J_z_star
        
        # CHUẨN HÓA CỤC BỘ TỪNG ĐIỂM: Chia cho nu_scale (nu_star.detach())
        nu_scale = torch.clamp(nu_star.detach(), min=1e-6)
        return res_pde / nu_scale"""

    pattern = r"    def compute_linear_pde_residual\(self, xy, A_z_star, nu_phys, J_z, H_cx, H_cy, nu_star_net=None\):.*?(?=\n    def |\Z)"
    if re.search(pattern, content, flags=re.DOTALL):
        updated = re.sub(pattern, new_fn + "\n\n", content, flags=re.DOTALL)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f"[✔] Đã cập nhật chuẩn hóa điểm PDE residual trong: {file_path}")

def update_compute_loss(base_dir):
    """2. Cân bằng tự động trọng số miền vật liệu (Domain-level Scaling) trong compute_loss.py."""
    file_path = os.path.join(base_dir, "training_engine", "compute_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_loop = """    for key, data in domains_dict.items():
        domain = data["domain_object"]
        net_key = resolve_domain_net_key(domain, model)
        pts = data["interior_points"].to(device)
        
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)
                
            A_z_star = model.forward_network(net_key, pts)
            
            if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                nu_star_pred = model.forward_nu_network(net_key, pts)
                nu_star_net = nu_star_pred if nu_star_pred is not None else None
                nu_effective = nu_star_pred
            else:
                nu_star_net = None
                nu_phys_eval = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
                nu_effective = nu_phys_eval / pinn.simulation_domain_engine_instance.nu0

            nu_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
            hx, hy = domain.evaluate_magnetization_vector(pts)
            hx, hy = hx.to(device), hy.to(device)
            jz = domain.evaluate_current_density(pts).to(device)
            
            res_pde = training_manager_instance.pde_evaluator.compute_linear_pde_residual(
                xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_star_net=nu_star_net
            )
            fading_w = training_manager_instance.pde_evaluator.compute_fading_weight(domain, pts)
            
            loss_pde_domain = torch.mean((res_pde ** 2) * fading_w)
            
            # CÂN BẰNG TỰ ĐỘNG THEO ĐỘ TỪ THẨM TRUNG BÌNH MIỀN (Domain-level Scaling)
            with torch.no_grad():
                mean_nu_star = torch.mean(nu_effective).item()
                auto_domain_weight = float(np.clip(1.0 / max(mean_nu_star, 1e-6), 1.0, 10000.0))

            user_base_weight = float(getattr(domain.training_config, 'pde_weight', 1.0))
            effective_pde_weight = user_base_weight * auto_domain_weight

            scaled_pde_loss = effective_pde_weight * loss_pde_domain
            total_loss = total_loss + scaled_pde_loss
            pde_loss_sum += scaled_pde_loss.item()"""

    pattern = r"    for key, data in domains_dict\.items\(\):.*?pde_loss_sum \+= scaled_pde_loss\.item\(\)"
    if re.search(pattern, content, flags=re.DOTALL):
        updated = re.sub(pattern, new_loop, content, flags=re.DOTALL)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f"[✔] Đã cập nhật Cân bằng trọng số miền tự động trong: {file_path}")

def update_save_frame(base_dir):
    """3. Cập nhật truyền nu_star_net trong save_frame.py để đồng nhất hiển thị PDE Residual Error."""
    file_path = os.path.join(base_dir, "training_visualizer", "save_frame.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_block = """            if getattr(domain.physics_config, 'is_nonlinear', False):
                if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                    with torch.no_grad():
                        nu_star_pred = model.forward_nu_network(net_key, eval_points)
                        nu_dom = nu_star_pred * sim_domain.nu0
                        nu_star_net_eval = nu_star_pred
                else:
                    B_mag_sub = torch.sqrt(Bx_sub_phys**2 + By_sub_phys**2 + 1e-12)
                    nu_dom = domain.evaluate_reluctivity(eval_points, B_magnitude_tensor=B_mag_sub, is_nonlinear=True).to(computation_device)
                    nu_star_net_eval = None
            else:
                nu_dom = domain.evaluate_reluctivity(eval_points, is_nonlinear=False).to(computation_device)
                nu_star_net_eval = None

            hx_dom, hy_dom = domain.evaluate_magnetization_vector(eval_points)
            hx_dom, hy_dom = hx_dom.to(computation_device), hy_dom.to(computation_device)
            jz_dom = domain.evaluate_current_density(eval_points).to(computation_device)

            res_pde = pde_evaluator.compute_linear_pde_residual(
                eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom, nu_star_net=nu_star_net_eval
            )"""

    old_pattern = r"            if getattr\(domain\.physics_config, 'is_nonlinear', False\):.*?res_pde = pde_evaluator\.compute_linear_pde_residual\(\n                eval_points, Az_sub_star, nu_dom, jz_dom, hx_dom, hy_dom\n            \)"
    if re.search(old_pattern, content, flags=re.DOTALL):
        updated = re.sub(old_pattern, new_block, content, flags=re.DOTALL)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f"[✔] Đã cập nhật truyền nu_star_net trong: {file_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Thực thi sửa đổi cấu hình dự án tại: {base_dir}\n")
    
    update_maxwell_pde_loss(base_dir)
    update_compute_loss(base_dir)
    update_save_frame(base_dir)
    
    print("\n[✔] HOÀN TẤT TẤT CẢ SỬA ĐỔI! Tự động chuẩn hóa PDE Residual theo độ từ thẩm đã sẵn sàng.")

if __name__ == "__main__":
    main()