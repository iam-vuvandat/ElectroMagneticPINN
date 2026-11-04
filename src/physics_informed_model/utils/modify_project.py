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

def update_interface_loss(base_dir):
    """
    1. Cập nhật physics_domain/physical_equations/interface_loss.py:
       Tích hợp thuật toán Tự động Tính trọng số Giao diện theo Physical Impedance Ratio (Tỷ lệ bật nhảy từ thẩm jump ratio)
       và Cân bằng thành phần Az, Bn, Ht tự động.
    """
    file_path = os.path.join(base_dir, "physics_domain", "physical_equations", "interface_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_compute_pair = """    def _compute_pair_components(self, key_a, domain_a, key_b, domain_b, pts):
        if pts.shape[0] == 0:
            zero = torch.tensor(0.0, device=pts.device)
            return zero, zero, zero, zero

        model = self.electro_magnetic_pinn.pinn_architecture_instance
        nx, ny, tx, ty, corner_w = self._compute_normals_and_tangents_from_outline(domain_a, pts)

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            _, _, _, _, corner_w_b = self._compute_normals_and_tangents_from_outline(domain_b, pts)
            corner_w = torch.minimum(corner_w, corner_w_b)

        Az_a = model.forward_network(key_a, pts)
        Az_b = model.forward_network(key_b, pts)

        loss_val = torch.mean((Az_a - Az_b) ** 2)

        grad_a = torch.autograd.grad(
            outputs=Az_a,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_a),
            create_graph=True
        )[0]

        grad_b = torch.autograd.grad(
            outputs=Az_b,
            inputs=pts,
            grad_outputs=torch.ones_like(Az_b),
            create_graph=True
        )[0]

        Bx_a_star = grad_a[:, 1:2] * self.L0
        By_a_star = -grad_a[:, 0:1] * self.L0
        Bx_b_star = grad_b[:, 1:2] * self.L0
        By_b_star = -grad_b[:, 0:1] * self.L0

        Bn_a_star = Bx_a_star * nx + By_a_star * ny
        Bn_b_star = Bx_b_star * nx + By_b_star * ny
        loss_Bn = torch.mean(corner_w * ((Bn_a_star - Bn_b_star) ** 2))

        # Đánh giá nu* từ Net_nu hoặc vật liệu hiện hành
        if hasattr(model, 'nu_networks') and key_a in model.nu_networks:
            nu_a_star = model.forward_nu_network(key_a, pts)
        else:
            B_mag_a_phys = torch.sqrt(Bx_a_star**2 + By_a_star**2 + 1e-12) * (self.H0 / self.nu0)
            nu_a_star = domain_a.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_a_phys, is_nonlinear=True) / self.nu0

        hx_a, hy_a = domain_a.evaluate_magnetization_vector(pts)
        hx_a_star = hx_a / self.H0
        hy_a_star = hy_a / self.H0

        if domain_b is not None and getattr(domain_b, 'domain_name', '') != 'background':
            if hasattr(model, 'nu_networks') and key_b in model.nu_networks:
                nu_b_star = model.forward_nu_network(key_b, pts)
            else:
                B_mag_b_phys = torch.sqrt(Bx_b_star**2 + By_b_star**2 + 1e-12) * (self.H0 / self.nu0)
                nu_b_star = domain_b.evaluate_reluctivity(pts, B_magnitude_tensor=B_mag_b_phys, is_nonlinear=True) / self.nu0
            hx_b, hy_b = domain_b.evaluate_magnetization_vector(pts)
            hx_b_star = hx_b / self.H0
            hy_b_star = hy_b / self.H0
        else:
            nu_b_star = torch.ones_like(nu_a_star)
            hx_b_star = torch.zeros_like(hx_a_star)
            hy_b_star = torch.zeros_like(hy_a_star)

        Hx_a_star = nu_a_star * Bx_a_star - hx_a_star
        Hy_a_star = nu_a_star * By_a_star - hy_a_star
        Hx_b_star = nu_b_star * Bx_b_star - hx_b_star
        Hy_b_star = nu_b_star * By_b_star - hy_b_star

        Ht_a_star = Hx_a_star * tx + Hy_a_star * ty
        Ht_b_star = Hx_b_star * tx + Hy_b_star * ty
        nu_pair_scale = torch.clamp(torch.maximum(nu_a_star.detach(), nu_b_star.detach()), min=1e-6)
        ht_diff_normalized = (Ht_a_star - Ht_b_star) / nu_pair_scale
        loss_Ht = torch.mean(corner_w * (ht_diff_normalized ** 2))

        # --- THUẬT TOÁN TÍNH TRỌNG SỐ GIAO DIỆN TỰ ĐỘNG THÔNG MINH (Physical Impedance & Jump Scaling) ---
        with torch.no_grad():
            mean_nu_a = torch.mean(nu_a_star).item()
            mean_nu_b = torch.mean(nu_b_star).item()
            mu_a = 1.0 / max(mean_nu_a, 1e-6)
            mu_b = 1.0 / max(mean_nu_b, 1e-6)
            
            # Tỷ lệ bật nhảy từ thẩm (Physical Permeability Jump Ratio)
            jump_ratio = float(max(mu_a / max(mu_b, 1e-6), mu_b / max(mu_a, 1e-6)))
            auto_interface_scale = float(min(max(jump_ratio, 1.0), 2000.0))

        user_w_az = float(getattr(domain_a.training_config, 'interface_potential_weight', 1000.0))
        user_w_bn = float(getattr(domain_a.training_config, 'interface_bn_weight', 500.0))
        user_w_ht = float(getattr(domain_a.training_config, 'interface_ht_weight', 500.0))

        # Nếu giữ cấu hình mặc định (hoặc đặt 1.0), hệ thống tự động kích hoạt auto_interface_scale
        w_az = auto_interface_scale if user_w_az in [1000.0, 1.0] else user_w_az
        w_bn = auto_interface_scale if user_w_bn in [500.0, 1.0] else user_w_bn
        w_ht = auto_interface_scale if user_w_ht in [500.0, 1.0] else user_w_ht

        weighted_interface = w_az * loss_val + w_bn * loss_Bn + w_ht * loss_Ht
        return weighted_interface, loss_val, loss_Bn, loss_Ht"""

    pattern = r"    def _compute_pair_components\(self, key_a, domain_a, key_b, domain_b, pts\):.*?(?=\n    def |\Z)"
    if re.search(pattern, content, flags=re.DOTALL):
        updated = re.sub(pattern, new_compute_pair + "\n\n", content, flags=re.DOTALL)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated)
        print(f"[✔] Đã cập nhật Thuật toán Trọng số Giao diện Tự động trong: {file_path}")

def update_compute_loss(base_dir):
    """
    2. Cập nhật training_engine/compute_loss.py:
       Tích hợp Co-Energy Variational Loss Scaling cho interior PDE.
    """
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
            
            # CHUẨN HÓA TRỌNG SỐ MIỀN THEO ĐỒNG NĂNG LƯỢNG TỪ (Co-Energy Scaling: W_domain = 1 / mean_nu_star = mean_mur)
            with torch.no_grad():
                mean_nu_star = torch.mean(nu_effective).item()
                raw_auto_weight = 1.0 / max(mean_nu_star, 1e-4)
                auto_domain_weight = float(min(max(raw_auto_weight, 1.0), 2000.0))

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
        print(f"[✔] Đã cập nhật Co-Energy Scaling trong: {file_path}")

def update_test_simulations(base_dir):
    """3. Cập nhật các file test_simulation.py và test_simulation2.py để kích hoạt cơ chế trọng số tự động (đặt pde_weight = 1.0, interface_weight = 1.0)."""
    for fname in ["test_simulation.py", "test_simulation2.py"]:
        file_path = os.path.join(base_dir, "tests", fname)
        if not os.path.exists(file_path):
            alt_path = os.path.join(base_dir, "src", "physics_informed_model", "tests", fname)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Đặt interface weights về 1.0 để tự động dùng Jump Ratio Scaling
        content = re.sub(r"iron_yoke\.training_config\.interface_potential_weight\s*=\s*[\d\.]+", "iron_yoke.training_config.interface_potential_weight = 1.0", content)
        content = re.sub(r"iron_yoke\.training_config\.interface_bn_weight\s*=\s*[\d\.]+", "iron_yoke.training_config.interface_bn_weight = 1.0", content)
        content = re.sub(r"iron_yoke\.training_config\.interface_ht_weight\s*=\s*[\d\.]+", "iron_yoke.training_config.interface_ht_weight = 1.0", content)
        content = re.sub(r"iron_yoke\.training_config\.pde_weight\s*=\s*[\d\.]+", "iron_yoke.training_config.pde_weight = 1.0", content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật trọng số tự động trong: {file_path}")

def main():
    base_dir = get_base_dir()
    print(f"[*] Thực thi cập nhật Thuật toán Trọng số Tự động cho PINN/XPINN tại: {base_dir}\n")
    
    update_interface_loss(base_dir)
    update_compute_loss(base_dir)
    update_test_simulations(base_dir)
    
    print("\n[✔] HOÀN TẤT! Thuật toán đặt trọng số thông minh đã được cập nhật vào dự án.")

if __name__ == "__main__":
    main()