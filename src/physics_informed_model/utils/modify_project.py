import os

def execute_gpu_vectorization_and_memory_fix():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    eval_file = os.path.join(project_root, 'geometry_engine', 'global_physical_properties_evaluation.py')
    eval_code = """import torch

VACUUM_RELUCTIVITY = 795774.715459

def evaluate_global_physical_properties(segments_list, points_tensor):
    number_of_points = points_tensor.shape[0]
    computation_device = points_tensor.device
    
    global_reluctivity_tensor = torch.full((number_of_points, 1), VACUUM_RELUCTIVITY, dtype=torch.float32, device=computation_device)
    global_coercive_field_x_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_coercive_field_y_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_current_density_z_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)
    global_material_classification_tensor = torch.zeros((number_of_points, 1), dtype=torch.float32, device=computation_device)

    if not segments_list:
        return {
            "reluctivity": global_reluctivity_tensor,
            "coercive_field_x": global_coercive_field_x_tensor,
            "coercive_field_y": global_coercive_field_y_tensor,
            "current_density_z": global_current_density_z_tensor,
            "material_classification": global_material_classification_tensor
        }

    sdfs = []
    masks = []
    reluctivities = []
    hxs, hys, jzs = [], [], []

    for segment_object in segments_list:
        sdf = segment_object.compute_signed_distance_field(points_tensor)
        sdfs.append(sdf)
        
        actual_k = 5000.0 - segment_object.steepness * 4960.0
        mask = torch.sigmoid(-actual_k * sdf).view(-1, 1)
        
        masks.append(mask)
        reluctivities.append(segment_object.evaluate_reluctivity(points_tensor))
        hx, hy = segment_object.evaluate_magnetization_vector(points_tensor)
        hxs.append(hx)
        hys.append(hy)
        jzs.append(segment_object.evaluate_current_density(points_tensor))

    stacked_sdfs = torch.stack(sdfs, dim=1).squeeze(-1)
    global_sdf, _ = torch.min(stacked_sdfs, dim=1, keepdim=True)
    
    max_steepness = max([seg.steepness for seg in segments_list])
    envelope_k = 5000.0 - max_steepness * 4960.0
    global_envelope_mask = torch.sigmoid(-envelope_k * global_sdf).view(-1, 1)

    stacked_masks = torch.stack(masks, dim=1)
    stacked_reluctivities = torch.stack(reluctivities, dim=1)
    stacked_hxs = torch.stack(hxs, dim=1)
    stacked_hys = torch.stack(hys, dim=1)
    stacked_jzs = torch.stack(jzs, dim=1)

    total_mask = torch.sum(stacked_masks, dim=1)
    safe_total_mask = total_mask + 1e-12 
    weights = stacked_masks / safe_total_mask.unsqueeze(1) 

    blended_reluctivity = torch.sum(weights * stacked_reluctivities, dim=1)
    blended_hx = torch.sum(weights * stacked_hxs, dim=1)
    blended_hy = torch.sum(weights * stacked_hys, dim=1)
    blended_jz = torch.sum(weights * stacked_jzs, dim=1)
    
    mat_indices = torch.arange(1.0, len(segments_list) + 1.0, device=computation_device).view(1, -1, 1)
    blended_mat = torch.sum(weights * mat_indices, dim=1)

    stacked_h_mag = torch.sqrt(stacked_hxs**2 + stacked_hys**2)
    target_h_mag = torch.sum(weights * stacked_h_mag, dim=1)

    blended_h_mag = torch.sqrt(blended_hx**2 + blended_hy**2)
    safe_blended_h_mag = torch.where(blended_h_mag < 1e-6, torch.full_like(blended_h_mag, 1.0), blended_h_mag)
    scale_factor = target_h_mag / safe_blended_h_mag
    
    has_field_mask = (target_h_mag > 1.0).float()
    blended_hx = blended_hx * (1.0 - has_field_mask) + (blended_hx * scale_factor) * has_field_mask
    blended_hy = blended_hy * (1.0 - has_field_mask) + (blended_hy * scale_factor) * has_field_mask

    global_reluctivity_tensor = VACUUM_RELUCTIVITY + global_envelope_mask * (blended_reluctivity - VACUUM_RELUCTIVITY)
    global_coercive_field_x_tensor = global_envelope_mask * blended_hx
    global_coercive_field_y_tensor = global_envelope_mask * blended_hy
    global_current_density_z_tensor = global_envelope_mask * blended_jz
    global_material_classification_tensor = global_envelope_mask * blended_mat

    return {
        "reluctivity": global_reluctivity_tensor,
        "coercive_field_x": global_coercive_field_x_tensor,
        "coercive_field_y": global_coercive_field_y_tensor,
        "current_density_z": global_current_density_z_tensor,
        "material_classification": global_material_classification_tensor
    }
"""

    train_file = os.path.join(project_root, 'training_manager.py')
    train_code = """import torch
import torch.optim as optim

class TrainingManager:
    def __init__(
        self, 
        model, 
        pde_evaluator, 
        lr_adam=1e-3, 
        target_loss=0.0,
        lbfgs_lr=0.8, 
        lbfgs_max_iter=1000, 
        lbfgs_max_eval=1250,
        lbfgs_tolerance_grad=1e-8,
        lbfgs_tolerance_change=1e-10,
        lbfgs_history_size=50
    ):
        self.model = model
        self.pde_evaluator = pde_evaluator
        
        self.base_lr_adam = lr_adam
        self.target_loss = target_loss
        self.lbfgs_lr = lbfgs_lr
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval
        self.lbfgs_tolerance_grad = lbfgs_tolerance_grad
        self.lbfgs_tolerance_change = lbfgs_tolerance_change
        self.lbfgs_history_size = lbfgs_history_size
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_lr_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_lr,
            max_iter=self.lbfgs_max_iter,
            max_eval=self.lbfgs_max_eval,
            tolerance_grad=self.lbfgs_tolerance_grad,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )

    def compute_loss(self, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        A_z_star = self.model(points_tensor)
        residual_star = self.pde_evaluator.compute_residual(
            xy=points_tensor, A_z_star=A_z_star, nu=reluctivity_tensor,
            J_z=current_density_z_tensor, H_cx=coercive_field_x_tensor, H_cy=coercive_field_y_tensor
        )
        return torch.mean(residual_star**2)

    def train_adam(self, epochs, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        self.model.train()
        best_loss = float('inf')
        best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
        
        for param_group in self.optimizer_adam.param_groups:
            param_group['initial_lr'] = self.base_lr_adam
            param_group['lr'] = self.base_lr_adam
            
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(epochs):
            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_loss(points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor)
            
            if torch.isnan(loss) or loss.item() > 1.5 * best_loss:
                self.model.load_state_dict(best_model_state)
                for param_group in self.optimizer_adam.param_groups:
                    param_group['lr'] *= 0.8
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            current_loss_value = loss.item()
            if current_loss_value < best_loss:
                best_loss = current_loss_value
                best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                
            if self.target_loss > 0 and current_loss_value <= self.target_loss:
                print(f"Adam Epoch {epoch + 1}: Đạt ngưỡng target_loss. KẾT THÚC ADAM SỚM!")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Adam Epoch {epoch + 1}: Loss = {current_loss_value:.6e} | LR = {self.optimizer_adam.param_groups[0]['lr']:.3e}")

    def train_lbfgs(self, epochs, points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor):
        self.model.train()
        lbfgs_counter = [0]
        early_stop_triggered = False 
        
        def closure():
            nonlocal early_stop_triggered
            self.optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = self.compute_loss(points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor)
            loss.backward()
            
            lbfgs_counter[0] += 1
            if lbfgs_counter[0] == 1 or lbfgs_counter[0] % 20 == 0:
                print(f"L-BFGS Step {lbfgs_counter[0]}: Loss = {loss.item():.6e}")
                
            if self.target_loss > 0 and loss.item() <= self.target_loss:
                early_stop_triggered = True
            return loss
            
        for epoch in range(epochs):
            if early_stop_triggered:
                print(f"L-BFGS Epoch {epoch + 1}: Đạt ngưỡng target_loss. KẾT THÚC L-BFGS SỚM!")
                break
            self.optimizer_lbfgs.step(closure)
"""

    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(eval_code)
        
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_code)

if __name__ == '__main__':
    execute_gpu_vectorization_and_memory_fix()