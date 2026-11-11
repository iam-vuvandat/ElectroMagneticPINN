import matplotlib.pyplot as plt
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

    domain_colors = ['#1a365d', '#742a2a', '#1c4532', '#4a5568', '#44337a', '#7b341e']
    pair_colors = ['#800000', '#004080', '#004d40', '#4a148c', '#bf360c']

    domains_dict = domains_data.get("domains", {})
    shared_dict = domains_data.get("shared_interfaces", {})

    for idx, (domain_key, data) in enumerate(domains_dict.items()):
        domain_obj = data.get("domain_object")
        d_name = getattr(domain_obj, "domain_name", domain_key)
        c_dom = domain_colors[idx % len(domain_colors)]
        
        interior_pts = data.get("interior_points")
        surface_pts = data.get("outer_surface_points", data.get("surface_points"))

        if interior_pts is not None and interior_pts.shape[0] > 0:
            if hasattr(interior_pts, "cpu"):
                interior_pts = interior_pts.cpu().detach().numpy()
            ax.scatter(
                interior_pts[:, 0], interior_pts[:, 1], 
                s=0.5, color=c_dom, marker='o', linewidths=0, alpha=0.55
            )

        if surface_pts is not None and surface_pts.shape[0] > 0:
            if hasattr(surface_pts, "cpu"):
                surface_pts = surface_pts.cpu().detach().numpy()
            ax.scatter(
                surface_pts[:, 0], surface_pts[:, 1], 
                s=1.5, color=c_dom, marker='o', linewidths=0, alpha=0.75
            )

    for p_idx, (pair_key, pair_data) in enumerate(shared_dict.items()):
        if_pts = pair_data.get("interface_points")
        if if_pts is not None and if_pts.shape[0] > 0:
            if hasattr(if_pts, "cpu"):
                if_pts = if_pts.cpu().detach().numpy()
            c_pair = pair_colors[p_idx % len(pair_colors)]
            ax.scatter(
                if_pts[:, 0], if_pts[:, 1], 
                s=1.0, color=c_pair, marker='o', linewidths=0, alpha=0.95
            )

    ax.set_title("Categorized Collocation Points", fontsize=13, fontweight="bold")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[*] Saved categorized collocation points plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax