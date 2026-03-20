#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment  
import os
import csv
import time
import pandas as pd

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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
        # 🛡️ [SRM 模块引入]: 安全返航状态机
        # 对比 baseline: 彻底重构了系统的生命周期，增加了统一的返航时间、初始/目标点快照。
        # =================================================================
        self.is_returning = False
        self.return_start_poses = None
        self.return_home_poses = None
        self.return_start_time = 0
        self.return_duration = 5.0
        
        # =================================================================
        # 🚀 [FMS 模块引入]: 动态活跃数量追踪
        # 对比 baseline: 避免了全局遍历，使得天地飞机的状态得以解耦分离。
        # =================================================================
        self.current_shape_num = 0
        self.current_active_num = 0
        self.moving_mask = None 

        # 🌟 新增：FMS 可视化与精确状态探针
        self.fms_dir = ""
        self.phase_start_time = 0.0
        self.trajectory_log = []
        self.frame_counter = 0
        self.drone_states = np.zeros(1000) # 记录分配目标：1=去往形状(星标), 2=去往基地(灰点)

    # =================================================================
    # 🛡️ [SRM 模块引入]: 核心返航初始化函数
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
            
            dist_matrix = cdist(self.return_start_poses[:m], self.return_home_poses[:m])
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched_homes = np.zeros_like(self.return_home_poses[:m])
            for r, c in zip(row_ind, col_ind):
                matched_homes[r] = self.return_home_poses[:m][c].copy()
            self.return_home_poses[:m] = matched_homes
            
            max_dist = np.max(np.linalg.norm(self.return_home_poses[:m] - self.return_start_poses[:m], axis=1))
        else:
            max_dist = 0
            
        self.return_duration = max(max_dist / (self.max_vel * 0.45), 6.0) 
        
        self.return_start_time = time.time()
        self.goals = self.return_start_poses.copy() 
        
        print(f"\n[SRM] Safe Return Activated. {m} active drones returning. Est. time: {self.return_duration:.1f}s")

    # =================================================================
    # 🧠 [DCA + FMS 混合模块]: 重构的目标分配器
    # =================================================================
    def distribute_goals(self, start, goals, shape_num=None, active_num=None):
        if shape_num is None: shape_num = len(goals)
        if active_num is None: active_num = len(goals)
        
        self.current_shape_num = shape_num
        self.current_active_num = active_num

        out_goals = np.copy(goals)

        if active_num > 0:
            active_start = start[:active_num]
            active_goals = goals[:active_num]
            shape_goals = active_goals[:shape_num]
            rtb_goals = active_goals[shape_num:]

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

                if len(rtb_goals) > 0:
                    scaled_active_goals = np.vstack((scaled_shape_goals, rtb_goals))
                else:
                    scaled_active_goals = scaled_shape_goals

                cost_matrix = cdist(active_start, scaled_active_goals) ** 2.0 
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                self.drone_states.fill(0)
                for r, c in zip(row_ind, col_ind):
                    out_goals[r] = scaled_active_goals[c]
                    self.drone_states[r] = 1 if c < shape_num else 2 # 🌟 1=飞往形状, 2=返航基地
                print(f"\n[DCA] Dynamic Configuration Assignment (Scale: {scale:.2f}). Shape drones: {shape_num}")
                
            else:
                dist_matrix = cdist(active_start, active_goals)
                self.drone_states.fill(0)
                for i in range(active_num):
                    ind = np.argmin(dist_matrix[i])
                    out_goals[i] = active_goals[ind]
                    self.drone_states[i] = 1 if ind < shape_num else 2 # 🌟 1=飞往形状, 2=返航基地
                    dist_matrix[i, :] = np.inf
                    dist_matrix[:, ind] = np.inf
                print(f"\n[Baseline] Greedy topology. Shape drones: {shape_num}")

        self.goals = out_goals

    def get_control(self, poses) -> None:
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
        
        for k in range(len(control_vels)):
            speed = np.linalg.norm(control_vels[k])
            if speed > self.max_vel:
                control_vels[k] = (control_vels[k] / speed) * self.max_vel
        self.velocities[:n] = control_vels.copy()

        # 🌟 新增：精确捕捉每一帧的动态流
        self.frame_counter += 1
        if self.frame_counter % 10 == 0 and hasattr(self, 'fms_dir') and self.fms_dir:
            current_t = time.time() - self.phase_start_time
            
            # 极简防撞计算
            active_poses = poses[:self.current_active_num]
            if self.current_active_num > 1:
                dists = np.linalg.norm(active_poses[:, np.newaxis, :] - active_poses[np.newaxis, :, :], axis=-1)
                np.fill_diagonal(dists, np.inf)
                phase_min_dist = np.min(dists)
            else:
                phase_min_dist = self.min_dist

            for i in range(len(poses)):
                state = 0 # 0=躺平(IDLE)
                if self.is_returning:
                    if getattr(self, 'moving_mask', np.ones(n, dtype=bool))[i]: state = 2 # 返航
                elif i < self.current_active_num:
                    state = self.drone_states[i] # 获取这架飞机本次的任务：1=构图，2=被踢出返航

                if state != 0 or poses[i][2] > 0.1: 
                    self.trajectory_log.append([current_t, i, poses[i][0], poses[i][1], poses[i][2], state, phase_min_dist])

        if self.log_dir and self.current_log_name and self.current_shape_num > 0:
            full_path = os.path.join(self.log_dir, f"{self.current_log_name}.csv")
            if self.last_csv_path != full_path:
                self.csv_initialized = False
                self.last_csv_path = full_path

            if not self.csv_initialized:
                try:
                    with open(full_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Time(s)", "Min_Distance(m)", "Avg_Velocity(m/s)", "Target_Error(m)"])
                    self.start_time = time.time()
                    self.csv_initialized = True
                except Exception:
                    return control_vels

            curr_t = round(time.time() - self.start_time, 2)
            eval_poses = poses[:self.current_shape_num]
            eval_goals = self.goals[:self.current_shape_num]
            eval_vels = control_vels[:self.current_shape_num]
            
            if self.current_shape_num > 1:
                diffs = eval_poses[:, np.newaxis, :] - eval_poses[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=-1)
                np.fill_diagonal(dists, np.inf)
                min_d = round(np.min(dists), 4)
            else:
                min_d = 0.0
                
            avg_v = round(np.mean(np.linalg.norm(eval_vels, axis=1)), 4)
            err = round(np.mean(np.linalg.norm(eval_goals - eval_poses, axis=1)), 4)

            try:
                with open(full_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([curr_t, min_d, avg_v, err])
            except:
                pass 
        return control_vels

    def generate_plots(self):
        if not self.last_csv_path or not os.path.exists(self.last_csv_path): return
        mode_prefix = "DCA" if self.enable_dca else "Base"
        algo_label = "DCA (Ours)" if self.enable_dca else "Baseline"

        print(f"\n[*] Generating plots for [{algo_label}] mode...")
        try:
            df = pd.read_csv(self.last_csv_path)
            metrics = {
                'Target_Error(m)': ('Convergence Error Comparison', 'Mean Error (m)', '#2ECC71' if self.enable_dca else '#E74C3C'),
                'Min_Distance(m)': ('Minimum Distance Comparison', 'Min Distance (m)', '#2ECC71' if self.enable_dca else '#E74C3C'),
                'Avg_Velocity(m/s)': ('Average Velocity Comparison', 'Avg Velocity (m/s)', '#2ECC71' if self.enable_dca else '#E74C3C')
            }
            for col, (title, ylabel, color) in metrics.items():
                if col in df.columns:
                    plt.figure(figsize=(9, 5.5))
                    plt.plot(df['Time(s)'], df[col], linewidth=2.5 if self.enable_dca else 1.5, 
                             color=color, linestyle='-' if self.enable_dca else '--', 
                             label=algo_label, alpha=0.9)
                    
                    if col == 'Target_Error(m)':
                        plt.axhline(y=0.0, color='black', linestyle=':', label='Ideal')
                    elif col == 'Min_Distance(m)':
                        # 这里动态使用了 self.min_dist，实现了图表随用户输入变化
                        plt.axhline(y=self.min_dist, color='black', linestyle='-.', label=f'Safety Limit ({self.min_dist}m)')
                        plt.axhspan(0, self.min_dist, color='gray', alpha=0.15)
                        plt.ylim(bottom=max(0, self.min_dist - 0.05), top=df[col].max() * 1.05)
                    elif col == 'Avg_Velocity(m/s)':
                        plt.axhline(y=self.max_vel, color='blue', linestyle=':', alpha=0.5, label='Max Velocity')
                        plt.ylim(bottom=-0.05, top=self.max_vel + 0.1)

                    plt.title(title, fontweight='bold', fontsize=14)
                    plt.xlabel('Time $t$ (s)', fontsize=12)
                    plt.ylabel(ylabel, fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
                    
                    img_name = f"{mode_prefix}_{self.current_log_name}_{col.split('(')[0]}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.log_dir, img_name), dpi=300)
                    plt.close()
            print(f"[*] Plots saved: {self.log_dir}")
        except Exception as e:
            print(f"⚠️ Plotting Error: {e}")

    def generate_idle_report(self, home_poses):
        """🌟 第一张图：初始纯灰点待命图"""
        if not hasattr(self, 'fms_dir') or not self.fms_dir: return
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Phase 0: FMS Fleet Standby", fontweight='bold')
        # 画出全局停机坪的灰色锚点
        ax.scatter(home_poses[:,0], home_poses[:,1], home_poses[:,2], color='gray', marker='o', s=30, label='Idle Fleet')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.fms_dir, "00_FMS_Standby.png"), dpi=300)
        plt.close()

    def generate_fms_srm_report(self, phase_name, phase_idx):
        """🌟 阶段图：支持星标与灰点切换，轨迹流线"""
        if not self.trajectory_log or not hasattr(self, 'fms_dir') or not self.fms_dir: return
        print(f"\n[*] 📊 [FMS] Generating Trajectory Map for Phase {phase_idx}...")
        df = pd.DataFrame(self.trajectory_log, columns=['Time', 'DroneID', 'X', 'Y', 'Z', 'State', 'MinDist'])
        
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f"Phase {phase_idx}: [{phase_name}] Trajectory & FMS Status", fontweight='bold')
        
        # 为了提供对比参照，先铺一层底层基地的灰色圆点
        if hasattr(self, 'global_home_poses') and self.global_home_poses is not None:
            ax1.scatter(self.global_home_poses[:,0], self.global_home_poses[:,1], self.global_home_poses[:,2], 
                        color='lightgray', marker='o', s=20, alpha=0.5, label='Base Grid')
        
        plotted = set()
        for drone_id in df['DroneID'].unique():
            drone_data = df[df['DroneID'] == drone_id]
            states = drone_data['State'].unique()
            
            # == 状态 1：被 FMS 征召去构成形状 ==
            if 1 in states: 
                line = drone_data[drone_data['State'] == 1]
                lbl1 = "FMS Dispatch" if "Disp" not in plotted else ""
                ax1.plot(line['X'], line['Y'], line['Z'], color='#3498DB', alpha=0.5, label=lbl1)
                # 终点标记：高亮的绿色星型 🌟
                end_pt = line.iloc[-1]
                lbl_star = "Shape Node" if "Star" not in plotted else ""
                ax1.scatter(end_pt['X'], end_pt['Y'], end_pt['Z'], color='#2ECC71', marker='*', s=150, zorder=5, label=lbl_star)
                plotted.update(["Disp", "Star"])
            
            # == 状态 2：被 FMS 踢出或触发 SRM 返航 ==
            if 2 in states: 
                line = drone_data[drone_data['State'] == 2]
                lbl2 = "SRM Return" if "Ret" not in plotted else ""
                ax1.plot(line['X'], line['Y'], line['Z'], color='#E74C3C', alpha=0.5, linestyle=':', label=lbl2)
                # 终点标记：回到地面的灰色点 ⚪
                end_pt = line.iloc[-1]
                lbl_dot = "Landed/Idle" if "Dot" not in plotted else ""
                ax1.scatter(end_pt['X'], end_pt['Y'], end_pt['Z'], color='gray', marker='o', s=50, zorder=5, label=lbl_dot)
                plotted.update(["Ret", "Dot"])

        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
        ax1.legend()

        # === 子图 2：全程防撞铁证 ===
        ax2 = fig.add_subplot(122)
        ax2.set_title(f"Phase {phase_idx}: Zero-Collision Proof", fontweight='bold')
        time_dist = df.groupby('Time')['MinDist'].min().reset_index()
        ax2.plot(time_dist['Time'], time_dist['MinDist'], color='#2ECC71', linewidth=2, label='Min Inter-Drone Dist')
        
        col_rad = 0.2
        ax2.axhline(y=col_rad, color='red', linestyle='--', linewidth=2, label=f'Collision ({col_rad}m)')
        ax2.fill_between(time_dist['Time'], 0, col_rad, color='red', alpha=0.15)
        ax2.axhline(y=self.min_dist, color='gray', linestyle='-.', label=f'Baseline ({self.min_dist}m)')
        ax2.set_ylim(bottom=0.0)
        ax2.set_xlabel('Phase Time (s)'); ax2.set_ylabel('Distance (m)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fms_dir, f"{phase_idx:02d}_{phase_name}.png"), dpi=300)
        plt.close()