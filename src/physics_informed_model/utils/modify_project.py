import os

def execute_backward_graph_fix():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
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
                
            # KHÔI PHỤC RETAIN_GRAPH ĐỂ GIỮ SỐNG ĐỒ THỊ HÌNH HỌC TĨNH
            loss.backward(retain_graph=True)
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
            
            # KHÔI PHỤC RETAIN_GRAPH
            loss.backward(retain_graph=True)
            
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

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_code)

if __name__ == '__main__':
    execute_backward_graph_fix()