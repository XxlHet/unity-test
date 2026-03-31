#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment  
import time

try:
    from apf_data_collector import collect_step_data
    from apf_plotter import generate_fms_srm_report, generate_idle_report, generate_plots
except ImportError:
    # Allow module import when executed from workspace root/package context.
    from scripts.apf_data_collector import collect_step_data
    from scripts.apf_plotter import generate_fms_srm_report, generate_idle_report, generate_plots

class APFSwarmController():
    def __init__(self, p_cohesion=1.0, p_seperation=1.0, p_alignment=1.0, max_vel=0.5, min_dist=0.35) -> None:
        self.swarm = None
        self.goals = None
        
        self.min_dist = min_dist
        self.max_vel = max_vel
        
        self.velocities = None
        self.p_separation = p_seperation
        self.p_cohesion = p_cohesion

        # =================================================================
        # 🧠 [DCA 模块引入]: 全局开关
        # 对比 baseline: 原版只有贪心算法，这里引入了基于 LLM 拓扑优化的状态标志。
        # =================================================================
        self.enable_dca = False  
        
        self.log_dir = ""            
        self.current_log_name = ""   
        self.csv_initialized = False 
        self.start_time = 0.0        
        self.last_csv_path = ""      

        # =================================================================
        # 🛡️ [FLSM 模块引入]: 安全返航状态机
        # 对比 baseline: 彻底重构了系统的生命周期，增加了统一的返航时间、初始/目标点快照。
        # =================================================================
        self.is_returning = False
        self.return_start_poses = None
        self.return_home_poses = None
        self.return_start_time = 0
        self.return_duration = 5.0
        self.return_home_tol = 0.08
        
        # =================================================================
        # 🚀 [FLSM 模块引入]: 动态活跃数量追踪
        # 对比 baseline: 避免了全局遍历，使得天地飞机的状态得以解耦分离。
        # =================================================================
        self.current_shape_num = 0
        self.current_active_num = 0
        self.moving_mask = None 

        # 🌟 新增：FLSM 可视化与精确状态探针
        self.fms_dir = ""
        self.phase_start_time = 0.0
        self.trajectory_log = []
        self.frame_counter = 0
        self.drone_states = np.zeros(1000) # 记录分配目标：1=去往形状(星标), 2=去往基地(灰点)
        self.phase_prev_active_num = 0
        self.phase_shape_num = 0
        self.phase_active_num = 0
        self.phase_new_launch_ids = np.array([], dtype=int)
        self.phase_return_ids = np.array([], dtype=int)

    # =================================================================
    # 🛡️ [FLSM 模块引入]: 核心返航初始化函数
    # =================================================================
    def initiate_safe_return(self, start_poses, home_poses):
        self.is_returning = True
        n = min(len(start_poses), len(home_poses))
        self.return_start_poses = start_poses[:n].copy()
        self.return_home_poses = home_poses[:n].copy()
        
        self.moving_mask = np.zeros(n, dtype=bool)
        m = min(self.current_active_num, n)
        
        if m > 0:
            self.moving_mask[:m] = True
            # 顺序编号 FLSM：按索引返回对应 home
            max_dist = np.max(np.linalg.norm(self.return_home_poses[:m] - self.return_start_poses[:m], axis=1))
        else:
            max_dist = 0
            
        self.return_duration = max(max_dist / (self.max_vel * 0.45), 6.0) 
        
        self.return_start_time = time.time()
        self.goals = self.return_start_poses.copy() 
        
        print(f"\n[FLSM] Safe Return Activated. {m} active drones returning by index. Est. time: {self.return_duration:.1f}s")

    # =================================================================
    # 🧠 [DCA + FLSM 混合模块]: 重构的目标分配器
    # =================================================================
    def distribute_goals(self, start, goals, shape_num=None, active_num=None):
        if shape_num is None: shape_num = len(goals)
        if active_num is None: active_num = len(goals)

        active_num = min(active_num, len(start), len(goals))
        shape_num = min(shape_num, active_num)
        
        self.current_shape_num = shape_num
        self.current_active_num = active_num

        out_goals = np.copy(goals)

        if active_num > 0:
            shape_start = start[:shape_num]
            shape_goals = goals[:shape_num]
            # 顺序增减编约束：返航组保持按索引回 home，不参与跨组重排
            rtb_start = start[shape_num:active_num]
            rtb_goals = goals[shape_num:active_num]
            self.drone_states.fill(0)

            if self.enable_dca and shape_num > 1:
                shape_goals = shape_goals + np.random.normal(0, 1e-3, shape_goals.shape) 
                
                dist_matrix_llm = cdist(shape_goals, shape_goals)
                np.fill_diagonal(dist_matrix_llm, np.inf) 
                
                min_dists = np.min(dist_matrix_llm, axis=1)
                effective_min = np.median(min_dists)
                effective_min = max(effective_min, 0.02) 
                
                # =================================================================
                # 终极炼丹参数 1: 恒定下压比例 (0.985)
                # 无论基线是 0.28 还是 0.32，目标永远下压 1.5%。
                # 这保证了所有无人机永远有一个微小的“向内钻”的冲动，这是产生微震荡的发动机。
                # =================================================================
                target_spacing = self.min_dist * 1.05  # 🔧 微调：给足呼吸空间
                
                # =================================================================
                # 终极炼丹参数 2: 动态体积界限 [0.45, 1.8]
                # =================================================================
                scale = np.clip(target_spacing / effective_min, 0.45, 3.5)  # 🔧 微调：放宽上限防拥挤
                
                centroid = np.mean(shape_goals, axis=0)
                scaled_shape_goals = centroid + (shape_goals - centroid) * scale

                for _ in range(60):
                    dists = cdist(scaled_shape_goals, scaled_shape_goals)
                    np.fill_diagonal(dists, np.inf)
                    min_dists = np.min(dists, axis=1)
                    if np.min(min_dists) >= target_spacing * 0.99: break
                        
                    displacement = np.zeros_like(scaled_shape_goals)
                    for i in range(len(scaled_shape_goals)):
                        mask = dists[i] < target_spacing
                        if np.any(mask):
                            vecs = scaled_shape_goals[i] - scaled_shape_goals[mask]
                            ds = dists[i][mask].reshape(-1, 1)
                            ds_safe = np.maximum(ds, 1e-4) 
                            
                            pushes = (vecs / ds_safe) * (target_spacing - ds) * 0.25
                            displacement[i] = np.sum(pushes, axis=0)
                    
                    displacement = np.clip(displacement, -0.08, 0.08)
                    scaled_shape_goals += displacement

                # 仅在表演组内部执行 DCA 匹配，避免“随机上/下场”
                cost_matrix = cdist(shape_start, scaled_shape_goals) ** 2.0 
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r, c in zip(row_ind, col_ind):
                    out_goals[r] = scaled_shape_goals[c]
                    self.drone_states[r] = 1 # 🌟 飞往形状

                for i in range(shape_num, active_num):
                    out_goals[i] = goals[i]
                    self.drone_states[i] = 2 # 🌟 顺序返航/待命组
                print(f"\n[DCA] Dynamic Configuration Assignment (Scale: {scale:.2f}). Shape drones: {shape_num}")
                
            else:
                # Baseline 也限定在表演组内匹配
                dist_matrix = cdist(shape_start, shape_goals)
                for i in range(shape_num):
                    ind = np.argmin(dist_matrix[i])
                    out_goals[i] = shape_goals[ind]
                    self.drone_states[i] = 1
                    dist_matrix[i, :] = np.inf
                    dist_matrix[:, ind] = np.inf

                for i in range(shape_num, active_num):
                    out_goals[i] = goals[i]
                    self.drone_states[i] = 2
                print(f"\n[Baseline] Greedy topology. Shape drones: {shape_num}")

        self.goals = out_goals

    def get_control(self, poses) -> None:
        # 🌟 新增：记录单步计算起始时间，用于评测算法实时性
        step_start_time = time.time()
        
        n = min(self.goals.shape[0], poses.shape[0])
        poses = poses[:n]
        if self.velocities is None:
            self.velocities = np.zeros_like(poses)
            
        if self.is_returning:
            elapsed = time.time() - self.return_start_time
            progress = min(elapsed / self.return_duration, 1.0)
            smooth_p = progress * progress * (3 - 2 * progress) 
            bloom_scale = 1.0 + 1.2 * np.sin(np.pi * smooth_p) 
            
            m = min(self.current_active_num, n)
            if m > 0:
                centroid = np.mean(self.return_home_poses[:m], axis=0)
                bloomed_home = centroid + (self.return_home_poses[:m] - centroid) * bloom_scale
                self.goals[:m] = self.return_start_poses[:m] + (bloomed_home - self.return_start_poses[:m]) * smooth_p

        # 🚀 彻底解开雷达的 2D 封印，让它变成全真 3D 雷达！
        ball_tree = BallTree(poses, metric='euclidean')
        control_vels = np.zeros_like(poses)

        self.goals = np.nan_to_num(self.goals)

        error_vec = self.goals[:n] - poses[:n]
        dist_to_goal = np.linalg.norm(error_vec, axis=1, keepdims=True)
        
        # =================================================================
        # 终极炼丹参数 3: 移除人工死区 (0.02m)
        # 将减速带压缩到 2cm，不干涉物理势场的博弈，保留震荡的高频波折感。
        # =================================================================
        scaling = np.where(dist_to_goal < 0.02, dist_to_goal / 0.02, 1.0) 
        vel_cohesion = self.p_cohesion * error_vec * scaling

        # 在 get_control 函数中：
        for i, pose in enumerate(poses):
            # 🚀 1. 删掉 [:2]，让算法感知完整 3D 坐标！
            query_pose = pose 
            v_nom = vel_cohesion[i].copy()
            if np.linalg.norm(v_nom) > self.max_vel:
                v_nom = (v_nom / np.linalg.norm(v_nom)) * self.max_vel

            interaction_radius = self.min_dist * 2.0
            nearest_ind = ball_tree.query_radius(query_pose.reshape(1, -1), interaction_radius)[0][1:]
            
            v_rep = np.zeros(3)
            for ind in nearest_ind:
                p_rel = pose - poses[ind]
                dist = np.linalg.norm(p_rel)
                
                if dist < self.min_dist:
                    safe_dist = max(dist, 0.01)
                    rep_strength = self.p_separation
                    if dist < self.min_dist * 1.0: 
                        penetration = self.min_dist / max(dist, 0.01)
                        dynamic_mult = np.clip(1.2 * (penetration ** 20), 1.2, 3.0)  # 🔧 微调：最高排斥力压至 3 倍
                        rep_strength *= dynamic_mult
                        
                    repulsive_mag = rep_strength * (1.0 / safe_dist - 1.0 / self.min_dist) / (safe_dist ** 2 + 0.01)
                    v_rep += repulsive_mag * (p_rel / safe_dist)
            
            # 🚀 2. 极其重要：把原本这里的 v_rep[2] = 0 删掉！让高度参与排斥！
            control_vels[i] = v_nom + v_rep

        if self.is_returning:
            mask = getattr(self, 'moving_mask', np.ones(n, dtype=bool))
            for i in range(n):
                if not mask[i]:
                    control_vels[i] = 0.0
        else:
            for i in range(n):
                if i >= self.current_active_num:
                    control_vels[i] = 0.0

        # =================================================================
        # 终极炼丹参数 5: 惯性动量配重 (0.70 / 0.30)
        # 将惯量权重从 20% 提升至 30%。
        # 这赋予了无人机更大的“物理惯性”。在碰到极硬的 20次方弹簧时，
        # 它们会因为惯性产生轻微的过冲（Overshoot）和回弹，从而拉出极具生命力的圆润波浪！
        # =================================================================
        control_vels = 0.90 * control_vels + 0.10 * self.velocities[:n]  # 🔧 微调：大幅削弱残余惯性

        # 惯性融合后再次清零非活动机体，防止地面机体被历史速度“带偏”
        if self.is_returning:
            mask = getattr(self, 'moving_mask', np.ones(n, dtype=bool))
            for i in range(n):
                if not mask[i]:
                    control_vels[i] = 0.0
                    self.velocities[i] = 0.0
        else:
            for i in range(n):
                if i >= self.current_active_num:
                    control_vels[i] = 0.0
                    self.velocities[i] = 0.0

        # 返航末段近点锁定，消除最后几厘米残余误差
        if self.is_returning:
            m = min(self.current_active_num, n)
            for i in range(m):
                if np.linalg.norm(poses[i] - self.return_home_poses[i]) < self.return_home_tol:
                    self.goals[i] = self.return_home_poses[i]
                    control_vels[i] = 0.0
                    self.velocities[i] = 0.0
        
        for k in range(len(control_vels)):
            speed = np.linalg.norm(control_vels[k])
            if speed > self.max_vel:
                control_vels[k] = (control_vels[k] / speed) * self.max_vel
        self.velocities[:n] = control_vels.copy()

        collect_step_data(self, poses, control_vels, n, step_start_time)
        return control_vels

    def generate_plots(self):
        generate_plots(self)

    def generate_idle_report(self, home_poses):
        """🌟 第一张图：初始纯灰点待命图"""
        generate_idle_report(self, home_poses)

    def generate_fms_srm_report(self, phase_name, phase_idx):
        """🌟 阶段图：支持星标与灰点切换，轨迹流线"""
        generate_fms_srm_report(self, phase_name, phase_idx)