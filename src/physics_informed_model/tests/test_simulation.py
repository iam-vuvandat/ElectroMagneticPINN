# --- FILE: tests/test_simulation_3.py ---

import os
import sys
import shutil
from types import SimpleNamespace

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from electro_magnetic_pinn import ElectroMagneticPINN


def run_high_precision_fpi_simulation_test():
    output_dir = "figure_high_precision_simulation_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("[*] KHỞI TẠO MÔ PHỎNG XPINN ĐỘ CHÍNH XÁC CAO (DÙNG MISH, LOGSCALE MUR & TINH CHỈNH CẤU TRÚC)...")

    model = ElectroMagneticPINN()
    sim_domain = model.simulation_domain_engine_instance

    # =========================================================================
    # 1. KHUNG TỌA ĐỘ VÀ QUY MÔ VẬT LÝ TOÀN CỤC
    # =========================================================================
    sim_domain.physics_config.scale_L0 = 0.12
    sim_domain.physics_config.scale_H0 = 960000.0
    sim_domain.physics_config.scale_nu0 = 795774.715459
    sim_domain.physics_config.allow_overlapping_domains = False
    sim_domain.x_boundaries_tuple = (-0.12, 0.12)
    sim_domain.y_boundaries_tuple = (-0.12, 0.12)

    # =========================================================================
    # 2. CẤU HÌNH LẤY MẪU ĐỘNG (COLLOCATION SAMPLER CONFIG)
    # =========================================================================
    model.sampler_config.dynamic_resampling = True
    model.sampler_config.resample_interval = 100
    model.sampler_config.resample_points_before_lbfgs_stage = True

    # =========================================================================
    # 3. CẤU HÌNH TIẾN TRÌNH HUẤN LUYỆN CHO ADAM HỘI TỤ SÂU
    # =========================================================================
    model.training_config.enable_linear_warmup_stage = True
    model.training_config.enable_nonlinear_adaptation_stage = True
    model.training_config.enable_lbfgs_refinement_stage = True

    # --- CẤU HÌNH SỐ EPOCH VÀ FPI ĐỒNG ĐỀU (1500 EPOCHS / BƯỚC) ---
    epochs_warmup_target = 1500
    fpi_iterations = 10
    epochs_per_fpi = 1500                     # Mỗi bước FPI chạy 1500 Adam epochs

    model.training_config.fpi_outer_iterations = fpi_iterations
    model.training_config.fpi_alpha_damping = 1.0     # Damping 1.0 cho FPI
    model.training_config.fpi_tolerance = 1e-6

    # Cấu hình tổng số Adam epochs trong training_config
    model.training_config.number_of_adam_epochs = epochs_warmup_target + (fpi_iterations * epochs_per_fpi)
    model.training_config.linear_warmup_epochs_ratio = epochs_warmup_target / model.training_config.number_of_adam_epochs

    # Tối ưu Adam để hội tụ cực sâu
    model.training_config.adam_learning_rate = 1e-3
    model.training_config.target_loss_threshold = 1e-7
    model.training_config.adam_eta_min = 1e-6
    model.training_config.grad_clip_max_norm = 1.0
    model.training_config.max_non_finite_streak = 5
    model.training_config.adam_log_interval = 100

    # Cấu hình huấn luyện cho Mạng phụ Net_nu
    model.training_config.pretrain_nu_epochs = 1000  # Pre-train Net_nu 1000 epochs
    model.training_config.pretrain_nu_lr = 5e-3
    model.training_config.fit_nu_epochs = 800        # Fit Net_nu 800 epochs ở mỗi bước FPI
    model.training_config.fit_nu_lr = 3e-3

    # Cấu hình L-BFGS Refinement tinh chỉnh sâu ở giai đoạn cuối
    model.training_config.number_of_lbfgs_iterations = 3000
    model.training_config.lbfgs_learning_rate = 0.5

    # =========================================================================
    # 4. CẤU HÌNH TRỰC QUAN HÓA (THANG LOGSCALE CHO MU_R TỪ 1.0 ĐẾN 7000.0)
    # =========================================================================
    model.visualization_config.active = True
    model.visualization_config.update_interval = 100
    model.visualization_config.output_directory = output_dir
    model.visualization_config.resolution = 160
    model.visualization_config.generate_gif_animation = True
    model.visualization_config.gif_filename = "high_precision_mur_training_process.gif"
    model.visualization_config.generate_mp4_video = True
    model.visualization_config.mp4_filename = "high_precision_mur_training_process.mp4"
    model.visualization_config.animation_fps = 5.0
    model.visualization_config.gif_fps = 5.0
    model.visualization_config.mp4_codec = "libx264"
    model.visualization_config.mp4_quality = 8
    model.visualization_config.contour_levels = 80

if __name__ == "__main__":
    run_high_precision_fpi_simulation_test()
   