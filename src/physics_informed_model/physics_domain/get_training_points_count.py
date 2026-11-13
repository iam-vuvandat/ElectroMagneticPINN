def get_training_points_count(collocation_sampler_instance, domains_data=None):
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
