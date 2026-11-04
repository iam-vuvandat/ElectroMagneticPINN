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
    """1. Loại bỏ việc chia nu_scale trong maxwell_pde_loss.py để không làm bùng nổ Gradient trong lõi sắt."""
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

        # Trả về Residual vật lý chuẩn của Maxwell (KHÔNG CHIA nu_scale)
        res_pde = dHy_dx_star - dHx_dy_star - J_z_star
        return res_pde"""

    pattern = r"    def compute_linear_pde_residual\(self, xy, A_z_star, nu_phys, J_z, H_cx, H_cy, nu_star_net=None\):.*?(?=\n    def |\Z)"
    if re.search(pattern, content, flags=re.DOTALL):
        updated = re.sub(pattern, new_fn + "\n\n", content, flags=re.DOTALL)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f"[✔] Đã cập nhật Maxwell PDE Residual (không chia nu_scale) trong: {file_path}")

def update_compute_loss(base_dir):
    """2. Dùng Soft Logarithmic Scaling trong compute_loss.py để điều hòa trọng số miền lõi sắt."""
    file_path = os.path.join(base_dir, "training_engine", "compute_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if "import numpy as np" not in content:
        content = "import numpy as np\n" + content

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
            
            # CÂN BẰNG TỰ ĐỘNG DẠNG LOGARITHM MỀM (Soft Logarithmic Scaling)
            with torch.no_grad():
                mean_nu_star = torch.mean(nu_effective).item()
                mean_mur = 1.0 / max(mean_nu_star, 1e-6)
                # Tăng nhẹ theo log1p(mur), giới hạn trong khoảng [1.0, 10.0]
                auto_domain_weight = float(min(max(1.0 + np.log1p(mean_mur), 1.0), 10.0))

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
        print(f"[✔] Đã cập nhật Soft Logarithmic Scaling trong: {file_path}")

def update_test_simulation(base_dir):
    """3. Cập nhật test_simulation2.py tăng cường liên tục giao diện và giữ pde_weight = 1.0."""
    file_path = os.path.join(base_dir, "tests", "test_simulation2.py")
    if not os.path.exists(file_path):
        alt_path = os.path.join(base_dir, "src", "physics_informed_model", "tests", "test_simulation2.py")
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"[!] Không tìm thấy file: {file_path}")
            return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Cập nhật interface_bn_weight, interface_ht_weight và pde_weight
    content = re.sub(r"iron_yoke\.training_config\.interface_bn_weight\s*=\s*[\d\.]+", "iron_yoke.training_config.interface_bn_weight = 2000.0", content)
    content = re.sub(r"iron_yoke\.training_config\.interface_ht_weight\s*=\s*[\d\.]+", "iron_yoke.training_config.interface_ht_weight = 2000.0", content)
    content = re.sub(r"iron_yoke\.training_config\.pde_weight\s*=\s*[\d\.]+", "iron_yoke.training_config.pde_weight = 1.0", content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[✔] Đã cập nhật interface weights cho lõi sắt trong: {file_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Thực thi cập nhật và khắc phục lỗi Flux Exclusion tại: {base_dir}\n")
    
    update_maxwell_pde_loss(base_dir)
    update_compute_loss(base_dir)
    update_test_simulation(base_dir)
    
    print("\n[✔] HOÀN TẤT TẤT CẢ CÁC CẬP NHẬT! Bạn có thể chạy lại test_simulation2.py.")

if __name__ == "__main__":
    main()