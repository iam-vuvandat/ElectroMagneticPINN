import os

files_to_create = {}

# 1. Tạo module vẽ đồ thị minh họa phân bố điểm lấy mẫu collocation
files_to_create["src/physics_informed_model/visualization/plot_collocation_sampling.py"] = '''import matplotlib.pyplot as plt
import torch


def plot_collocation_sampling(domains_data, save_path=None, show=True):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5
    })

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    domains_dict = domains_data.get("domains", {})
    for domain_key, data in domains_dict.items():
        domain_obj = data.get("domain_object")
        d_name = getattr(domain_obj, "domain_name", domain_key)
        
        interior_pts = data.get("interior_points")
        surface_pts = data.get("outer_surface_points", data.get("surface_points"))

        if interior_pts is not None and interior_pts.shape[0] > 0:
            if hasattr(interior_pts, "cpu"):
                interior_pts = interior_pts.cpu().detach().numpy()
            ax.scatter(interior_pts[:, 0], interior_pts[:, 1], s=4, alpha=0.5, label=f"{d_name} (Interior)")

        if surface_pts is not None and surface_pts.shape[0] > 0:
            if hasattr(surface_pts, "cpu"):
                surface_pts = surface_pts.cpu().detach().numpy()
            ax.scatter(surface_pts[:, 0], surface_pts[:, 1], s=10, alpha=0.8, label=f"{d_name} (Surface)")

    shared_dict = domains_data.get("shared_interfaces", {})
    for pair_key, pair_data in shared_dict.items():
        if_pts = pair_data.get("interface_points")
        if if_pts is not None and if_pts.shape[0] > 0:
            if hasattr(if_pts, "cpu"):
                if_pts = if_pts.cpu().detach().numpy()
            ax.scatter(if_pts[:, 0], if_pts[:, 1], s=12, alpha=0.9, label=f"Interface ({pair_key})")

    ax.set_title("Collocation Points Sampling Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("X Position [m]", fontsize=11)
    ax.set_ylabel("Y Position [m]", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
'''

# 2. Tạo module thống kê số lượng điểm collocation
files_to_create["src/physics_informed_model/physics_domain/get_training_points_count.py"] = '''def get_training_points_count(collocation_sampler_instance, domains_data=None):
    if domains_data is None:
        domains_data = collocation_sampler_instance.generate_all_domain_points()

    total_interior = 0
    total_surface = 0
    total_interface = 0

    domains_dict = domains_data.get("domains", {})
    for key, data in domains_dict.items():
        interior_pts = data.get("interior_points")
        surface_pts = data.get("outer_surface_points", data.get("surface_points"))
        if interior_pts is not None:
            total_interior += interior_pts.shape[0]
        if surface_pts is not None:
            total_surface += surface_pts.shape[0]

    shared_dict = domains_data.get("shared_interfaces", {})
    for pair_key, pair_data in shared_dict.items():
        if_pts = pair_data.get("interface_points")
        if if_pts is not None:
            total_interface += if_pts.shape[0]

    total_points = total_interior + total_surface + total_interface
    print(f"[*] Interior Points : {total_interior}")
    print(f"[*] Surface Points  : {total_surface}")
    print(f"[*] Interface Points: {total_interface}")
    print(f"[*] Total Training Points: {total_points}")

    return {
        "interior": total_interior,
        "surface": total_surface,
        "interface": total_interface,
        "total": total_points
    }
'''

def update_collocation_sampler():
    target_path = "src/physics_informed_model/physics_domain/collocation_sampler.py"
    if not os.path.exists(target_path):
        return

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()

    method_code = '''
    def get_training_points_count(self, domains_data=None):
        from physics_domain.get_training_points_count import get_training_points_count
        return get_training_points_count(self, domains_data=domains_data)
'''

    if "def get_training_points_count" not in content:
        content += "\n" + method_code
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[*] Updated: {target_path} (Added get_training_points_count)")

def update_electro_magnetic_pinn():
    target_path = "src/physics_informed_model/electro_magnetic_pinn.py"
    if not os.path.exists(target_path):
        return

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()

    methods_code = '''
    def visualize_collocation_sampling(self, domains_data=None, save_path=None, show=True):
        if domains_data is None:
            if self.collocation_sampler_instance is None:
                from physics_domain.collocation_sampler import CollocationSampler
                self.collocation_sampler_instance = CollocationSampler(
                    electro_magnetic_pinn=self,
                    device=getattr(self, 'computation_device', torch.device('cpu'))
                )
            domains_data = self.collocation_sampler_instance.generate_all_domain_points()

        from visualization.plot_collocation_sampling import plot_collocation_sampling
        return plot_collocation_sampling(domains_data=domains_data, save_path=save_path, show=show)

    def display_training_points_count(self, domains_data=None):
        if self.collocation_sampler_instance is None:
            from physics_domain.collocation_sampler import CollocationSampler
            self.collocation_sampler_instance = CollocationSampler(
                electro_magnetic_pinn=self,
                device=getattr(self, 'computation_device', torch.device('cpu'))
            )
        return self.collocation_sampler_instance.get_training_points_count(domains_data=domains_data)
'''

    if "def visualize_collocation_sampling" not in content:
        content += "\n" + methods_code
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[*] Updated: {target_path} (Added visualize_collocation_sampling & display_training_points_count)")

def main():
    for rel_path, file_content in files_to_create.items():
        dir_name = os.path.dirname(rel_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(rel_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        print(f"[*] Created/Updated: {rel_path}")

    update_collocation_sampler()
    update_electro_magnetic_pinn()
    print("[*] Update process completed successfully.")

if __name__ == "__main__":
    main()