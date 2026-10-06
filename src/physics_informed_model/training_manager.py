import torch
import torch.optim as optim

class TrainingManager:
    def __init__(
        self, 
        model, 
        pde_evaluator, 
        learning_rate_adam=1e-3, 
        target_loss=0.0,
        visualizer=None,
        visualizer_update_interval=100,
        resample_frequency=500,
        curriculum_ratio=0.25,
        loss_weight_uniform=1.0,
        loss_weight_interface=0.1
    ):
        self.model = model
        self.pde_evaluator = pde_evaluator
        self.target_loss = target_loss
        self.base_learning_rate_adam = learning_rate_adam
        
        self.resample_frequency = resample_frequency
        self.curriculum_ratio = curriculum_ratio
        self.loss_weight_uniform = loss_weight_uniform
        self.loss_weight_interface = loss_weight_interface
        
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=self.base_learning_rate_adam)
        
        self.visualizer = visualizer
        self.visualizer_update_interval = visualizer_update_interval

    def compute_dynamic_loss(self, pts_u, props_u, pts_i, props_i):
        # 1. Loss Vùng Nền
        A_z_u = self.model(pts_u)
        res_u = self.pde_evaluator.compute_residual(
            xy=pts_u, A_z_star=A_z_u, nu=props_u["reluctivity"],
            J_z=props_u["current_density_z"], H_cx=props_u["coercive_field_x"], H_cy=props_u["coercive_field_y"]
        )
        loss_u = torch.mean(res_u**2) * self.loss_weight_uniform

        # 2. Loss Ranh Giới (Được kìm hãm trọng số)
        A_z_i = self.model(pts_i)
        res_i = self.pde_evaluator.compute_residual(
            xy=pts_i, A_z_star=A_z_i, nu=props_i["reluctivity"],
            J_z=props_i["current_density_z"], H_cx=props_i["coercive_field_x"], H_cy=props_i["coercive_field_y"]
        )
        loss_i = torch.mean(res_i**2) * self.loss_weight_interface

        return loss_u + loss_i

    def _trigger_visualization(self, epoch, loss_val, pts_u, props_u, pts_i, props_i):
        if self.visualizer:
            comb_pts = torch.cat([pts_u, pts_i], dim=0)
            comb_nu = torch.cat([props_u['reluctivity'], props_i['reluctivity']], dim=0)
            comb_jz = torch.cat([props_u['current_density_z'], props_i['current_density_z']], dim=0)
            comb_hcx = torch.cat([props_u['coercive_field_x'], props_i['coercive_field_x']], dim=0)
            comb_hcy = torch.cat([props_u['coercive_field_y'], props_i['coercive_field_y']], dim=0)
            self.visualizer.save_frame(epoch, loss_val, comb_pts, comb_nu, comb_jz, comb_hcx, comb_hcy)

    def train_adam(self, epochs, sampler, geometry, sampler_config):
        self.model.train()
        best_loss = float('inf')
        best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
        
        num_u = int(sampler_config.number_of_uniform_points * self.curriculum_ratio)
        num_i = int(sampler_config.number_of_interface_points * self.curriculum_ratio)
        threshold = sampler_config.distance_threshold
        
        pts_u, props_u, pts_i, props_i = None, None, None, None
        
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adam, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(epochs):
            if epoch == 0 or (self.resample_frequency > 0 and epoch % self.resample_frequency == 0):
                pts_u = sampler.generate_uniform_points_tensor(num_u)
                pts_i = sampler.generate_interface_points_tensor(geometry, num_i, threshold)
                props_u = geometry.evaluate_global_physical_properties(pts_u)
                props_i = geometry.evaluate_global_physical_properties(pts_i)

            self.optimizer_adam.zero_grad(set_to_none=True)
            loss = self.compute_dynamic_loss(pts_u, props_u, pts_i, props_i)
            
            # PHÒNG THỦ: Kiểm tra NaN an toàn tuyệt đối
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1.5 * best_loss:
                self.model.load_state_dict(best_model_state)
                for param_group in self.optimizer_adam.param_groups:
                    param_group['lr'] *= 0.8
                continue
                
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer_adam.step()
            scheduler_adam.step()
            
            current_loss_value = loss.item()
            if current_loss_value < best_loss:
                best_loss = current_loss_value
                best_model_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
                
            if self.target_loss > 0 and current_loss_value <= self.target_loss:
                print(f"Adam Epoch {epoch + 1}: Đạt target_loss. KẾT THÚC SỚM!")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Adam Epoch {epoch + 1}: Loss = {current_loss_value:.6e} | LR = {self.optimizer_adam.param_groups[0]['lr']:.3e}")
                
            if (epoch + 1) % self.visualizer_update_interval == 0:
                self._trigger_visualization(epoch + 1, current_loss_value, pts_u, props_u, pts_i, props_i)
