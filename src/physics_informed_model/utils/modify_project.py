import os
import sys

def get_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..")),
        os.path.abspath(os.path.join(script_dir, "..", "..")),
        cwd,
        os.path.join(cwd, "src", "physics_informed_model"),
        "/content/ElectroMagneticPINN/src/physics_informed_model",
        "/content/ElectroMagneticPINN/physics_informed_model",
        "/content/ElectroMagneticPINN"
    ]
    
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "training_engine")) and os.path.isdir(os.path.join(cand, "architectures")):
            return cand
            
    return os.path.abspath(os.path.join(script_dir, ".."))


def fix_compute_loss(base_dir):
    """
    Sửa compute_loss.py: Chuẩn hóa thụt lề (Indentation) chuẩn 100% theo khối lệnh Python.
    """
    file_path = os.path.join(base_dir, "training_engine", "compute_loss.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    content = '''import torch
from training_engine.resolve_domain_net_key import resolve_domain_net_key

def compute_loss(training_manager_instance, domains_data):
    model = training_manager_instance.model
    device = next(model.parameters()).device
    dummy_param = next(model.parameters())
    total_loss = (dummy_param * 0.0).sum()

    pde_loss_sum = 0.0
    az_if_sum = 0.0
    bn_if_sum = 0.0
    ht_if_sum = 0.0
    weighted_if_sum = 0.0

    domains_dict = domains_data.get("domains", domains_data)
    shared_dict = domains_data.get("shared_interfaces", {})
    pinn = training_manager_instance.electro_magnetic_pinn
    bg_domain = pinn.simulation_domain_engine_instance.background_domain

    for key, data in domains_dict.items():
        domain = data["domain_object"]
        net_key = resolve_domain_net_key(domain, model)
        pts = data["interior_points"].to(device)
        
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)
                
            A_z_star = model.forward_network(net_key, pts)
            
            if hasattr(model, 'nu_networks') and net_key in model.nu_networks:
                nu_star_pred = model.forward_nu_network(net_key, pts)
                # Giu nguyen do thi autograd de tinh dao ham grad(nu) trong Maxwell PDE
                nu_star_net = nu_star_pred if nu_star_pred is not None else None
            else:
                nu_star_net = None

            nu_phys = domain.evaluate_reluctivity(pts, is_nonlinear=False).to(device)
            hx, hy = domain.evaluate_magnetization_vector(pts)
            hx, hy = hx.to(device), hy.to(device)
            jz = domain.evaluate_current_density(pts).to(device)
            
            res_pde = training_manager_instance.pde_evaluator.compute_linear_pde_residual(
                xy=pts, A_z_star=A_z_star, nu_phys=nu_phys, J_z=jz, H_cx=hx, H_cy=hy, nu_star_net=nu_star_net
            )
            fading_w = training_manager_instance.pde_evaluator.compute_fading_weight(domain, pts)
            
            loss_pde_domain = torch.mean((res_pde ** 2) * fading_w)
            w_pde = float(domain.training_config.pde_weight)

            scaled_pde_loss = w_pde * loss_pde_domain
            total_loss = total_loss + scaled_pde_loss
            pde_loss_sum += scaled_pde_loss.item()

        outer_pts = data.get("outer_surface_points", data.get("surface_points", torch.empty((0, 2)))).to(device)
        if outer_pts.shape[0] > 0 and net_key != 'background':
            if not outer_pts.requires_grad:
                outer_pts = outer_pts.detach().clone().requires_grad_(True)
                
            w_if_loss, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=net_key, domain_a=domain, key_b='background', domain_b=bg_domain, pts=outer_pts
            )
            scaled_if_loss = w_if_loss
            total_loss = total_loss + scaled_if_loss

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += scaled_if_loss.item()

    for pair_key, pair_data in shared_dict.items():
        pts = pair_data["interface_points"].to(device)
        if pts.shape[0] > 0:
            if not pts.requires_grad:
                pts = pts.detach().clone().requires_grad_(True)

            w_shared_loss, val_l, bn_l, ht_l = training_manager_instance.interface_evaluator._compute_pair_components(
                key_a=pair_data["key_a"], domain_a=pair_data["domain_a"],
                key_b=pair_data["key_b"], domain_b=pair_data["domain_b"], pts=pts
            )
            scaled_shared_loss = w_shared_loss
            total_loss = total_loss + scaled_shared_loss

            az_if_sum += val_l.item()
            bn_if_sum += bn_l.item()
            ht_if_sum += ht_l.item()
            weighted_if_sum += scaled_shared_loss.item()

    training_manager_instance.last_loss_breakdown = {
        "pde_loss": pde_loss_sum,
        "interface_Az": az_if_sum,
        "interface_Bn": bn_if_sum,
        "interface_Ht": ht_if_sum,
        "weighted_interface": weighted_if_sum,
        "total": total_loss.item()
    }

    return total_loss
'''
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[✔] Đã ghi đè file compute_loss.py với cấu trúc Indentation chuẩn xác tại: {file_path}")


def update_material_data_bh_curve(base_dir):
    """
    Cập nhật đường cong B-H thép từ mềm phi tuyến (mu_r_max ~ 6953 tại B = 0.9T).
    """
    file_path = os.path.join(base_dir, "simulation_domain_engine", "material_library", "material_data.py")
    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old_steel_block = """            SimpleNamespace(
                name="steel_1008",
                relative_permeability=1000.0,  
                coercive=[0.0, 0.0],
                bh_curve_data=[
                    # B_points (Tesla)
                    [0.0, 0.2402, 0.8654, 1.1106, 1.2458, 1.331, 1.5, 1.6, 1.683, 
                     1.741, 1.78, 1.905, 2.025, 2.085, 2.13, 2.165, 2.28, 2.485, 2.5851],
                    # H_points (A/m)
                    [0.0, 159.2, 318.3, 477.5, 636.6, 795.8, 1591.5, 3183.1, 4774.6, 
                     6366.2, 7957.7, 15915.5, 31831.0, 47746.5, 63662.0, 79577.5, 
                     159155.0, 318310.0, 397887.0]
                ]
            )"""

    new_steel_block = """            SimpleNamespace(
                name="steel_1008",
                relative_permeability=1000.0,  
                coercive=[0.0, 0.0],
                bh_curve_data=[
                    # B_points (Tesla) - Du lieu phi tuyen cao (mu_r_max = 6953.37 tai B = 0.9T)
                    [0.0, 0.1, 0.5, 0.8, 0.9, 1.0, 1.3, 1.5, 1.9, 2.4],
                    # H_points (A/m)
                    [0.0, 34.8, 67.4, 91.8, 103.0, 119.0, 250.0, 1180.0, 25000.0, 229580.0]
                ]
            )"""

    if old_steel_block in content:
        content = content.replace(old_steel_block, new_steel_block)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✔] Đã cập nhật bảng B-H phi tuyến cho steel_1008 tại: {file_path}")


def main():
    base_dir = get_base_dir()
    print(f"[*] Đang thực thi sửa đổi dự án tại: {base_dir}\n")
    
    fix_compute_loss(base_dir)
    update_material_data_bh_curve(base_dir)
    
    print("\n[✔] HOÀN TẤT SỬA LỖI INDENTATION VÀ CẬP NHẬT MÃ NGUỒN!")

if __name__ == "__main__":
    main()