import torch
import torch.optim as optim

class TrainingManager:
    def __init__(
        self, 
        model, 
        pde_evaluator, 
        learning_rate_adam=1e-3, 
        target_loss=0.0,
        lbfgs_learning_rate=0.8, 
        lbfgs_maximum_iterations=1000, 
        lbfgs_maximum_evaluations=1250,
        lbfgs_tolerance_gradient=1e-8,
        lbfgs_tolerance_change=1e-10,
        lbfgs_history_size=50,
        visualizer=None,
        visualizer_update_interval=100
    ):
        self.model = model
        self.pde_evaluator = pde_evaluator
        
        self.base_learning_rate_adam = learning_rate_adam
        self.target_loss = target_loss
        self.lbfgs_learning_rate = lbfgs_learning_rate
        self.lbfgs_maximum_iterations = lbfgs_maximum_iterations
        self.lbfgs_maximum_evaluations = lbfgs_maximum_evaluations
        self.lbfgs_tolerance_gradient = lbfgs_tolerance_gradient
        self.lbfgs_tolerance_change = lbfgs_tolerance_change
        self.lbfgs_history_size = lbfgs_history_size
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_learning_rate_adam)
        
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            lr=self.lbfgs_learning_rate,
            max_iter=self.lbfgs_maximum_iterations,
            max_eval=self.lbfgs_maximum_evaluations,
            tolerance_grad=self.lbfgs_tolerance_gradient,
            tolerance_change=self.lbfgs_tolerance_change,
            history_size=self.lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )
        
        self.visualizer = visualizer
        self.visualizer_update_interval = visualizer_update_interval

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
            param_group['initial_lr'] = self.base_learning_rate_adam
            param_group['lr'] = self.base_learning_rate_adam
            
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
                
            if self.visualizer and (epoch + 1) % self.visualizer_update_interval == 0:
                self.visualizer.save_frame(
                    epoch + 1, current_loss_value, 
                    points_tensor, reluctivity_tensor, current_density_z_tensor, coercive_field_x_tensor, coercive_field_y_tensor
                )

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
