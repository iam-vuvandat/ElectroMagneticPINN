def add_material_domain(self, domain_object):
    domain_object.simulation_domain = self
    if domain_object.outline_tensor is None:
        domain_object.build_domain()

    if getattr(domain_object, 'domain_name', '') != 'background':
        self.validate_non_overlapping_domains(candidate_domain=domain_object)

    if self.background_domain in self.domains_list:
        insert_idx = len(self.domains_list) - 1
        self.domains_list.insert(insert_idx, domain_object)
    else:
        self.domains_list.append(domain_object)
        
    for idx, dom in enumerate(self.domains_list):
        dom.domain_id = idx + 1
    return self
