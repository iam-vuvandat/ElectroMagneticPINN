def resolve_domain_net_key(domain, arch_instance):
    """Xác định net_key khớp với khóa đã đăng ký trong XPINNArchitecture."""
    d_name = getattr(domain, 'domain_name', '')
    if d_name == 'background':
        return 'background'

    d_id = getattr(domain, 'domain_id', None)
    candidates = []
    if d_id is not None:
        candidates.append(f'domain_{d_id}')
    if d_name:
        candidates.append(d_name)

    if arch_instance is not None:
        if hasattr(arch_instance, 'nu_networks'):
            for k in candidates:
                if k in arch_instance.nu_networks:
                    return k
        if hasattr(arch_instance, 'networks'):
            for k in candidates:
                if k in arch_instance.networks:
                    return k

    return candidates[0] if candidates else 'background'
