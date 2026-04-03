#!/usr/bin/env python3

import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree

try:
    from apf_data_collector import collect_step_data
    from apf_plotter import generate_fms_srm_report, generate_idle_report, generate_plots
except ImportError:
    from scripts.apf_data_collector import collect_step_data
    from scripts.apf_plotter import generate_fms_srm_report, generate_idle_report, generate_plots


class AFCAPFSwarmController:
    def __init__(self, max_vel=0.9, min_dist=0.35) -> None:
        self.swarm = None
        self.goals = None
        self.nominal_goals = None

        self.min_dist = float(min_dist)
        self.max_vel = float(max_vel)

        self.velocities = None
        self.enable_dca = False

        # APF terms.
        self.p_cohesion = 1.10
        self.p_separation = 1.0

        # AFC terms (affine + stress matrix feedback).
        self.k_affine = 0.08
        self.k_pressure = 0.12
        self.affine_smooth = 0.92
        self.shear_limit = 0.12
        self.scale_min = 0.85
        self.scale_max = 1.25
        self.fit_blend = 0.03
        self.center_blend = 0.01
        self.z_scale_gain = 0.75
        self.z_center_blend = 0.05
        self.z_vel_boost = 1.30
        self.nominal_anchor_gain = 0.96
        self.affine_fade_radius = 0.18
        self.max_affine_correction_ratio = 0.10
        self.near_goal_radius = 0.9
        self.near_goal_min_ratio = 0.03
        self.min_dist_stab_gain = 0.08
        self.min_dist_stab_ratio = 1.04
        self.affine_matrix = np.eye(3)
        self.stress_weights = None

        self.log_dir = ""
        self.current_log_name = ""
        self.csv_initialized = False
        self.start_time = 0.0
        self.last_csv_path = ""

        self.is_returning = False
        self.return_start_poses = None
        self.return_home_poses = None
        self.return_start_time = 0.0
        self.return_duration = 5.0
        self.return_home_tol = 0.08

        self.current_shape_num = 0
        self.current_active_num = 0
        self.moving_mask = None

        self.fms_dir = ""
        self.phase_start_time = 0.0
        self.trajectory_log = []
        self.frame_counter = 0
        self.drone_states = np.zeros(1000)
        self.phase_prev_active_num = 0
        self.phase_shape_num = 0
        self.phase_active_num = 0
        self.phase_new_launch_ids = np.array([], dtype=int)
        self.phase_return_ids = np.array([], dtype=int)
        self.global_home_poses = None

    def _build_stress_weights(self, shape_goals):
        n = len(shape_goals)
        if n <= 1:
            return np.zeros((n, n))

        d = cdist(shape_goals, shape_goals)
        np.fill_diagonal(d, np.inf)
        nn = np.minimum(6, n - 1)
        sigma = np.median(np.min(d, axis=1))
        sigma = max(float(sigma), 1e-3)

        w = np.zeros((n, n))
        for i in range(n):
            nbrs = np.argsort(d[i])[:nn]
            for j in nbrs:
                wij = np.exp(-((d[i, j] / sigma) ** 2))
                w[i, j] = max(w[i, j], wij)
                w[j, i] = max(w[j, i], wij)
        return w

    def _clip_affine_matrix(self, m):
        u, s, vh = np.linalg.svd(m)
        s = np.clip(s, self.scale_min, self.scale_max)
        m_sr = u @ np.diag(s) @ vh

        # Keep shear bounded for stability in dense swarms.
        m_clipped = m_sr.copy()
        for r in range(3):
            for c in range(3):
                if r != c:
                    m_clipped[r, c] = np.clip(m_clipped[r, c], -self.shear_limit, self.shear_limit)

        # Keep Z channel independent from XY affine deformation to preserve 3D altitude intent.
        m_clipped[0, 2] = 0.0
        m_clipped[1, 2] = 0.0
        m_clipped[2, 0] = 0.0
        m_clipped[2, 1] = 0.0
        m_clipped[2, 2] = np.clip(m_clipped[2, 2], 0.95, 1.15)
        return m_clipped

    def _estimate_pressure_scale(self, poses, shape_n):
        if shape_n <= 1:
            return 1.0
        d = cdist(poses[:shape_n], poses[:shape_n])
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d, axis=1)
        effective = max(float(np.median(min_pair)), 1e-3)
        size_scale = float(np.clip(np.sqrt(24.0 / max(shape_n, 1)), 0.60, 1.00))
        size_affine = size_scale * size_scale * size_scale
        if shape_n <= 30:
            # Small swarms should avoid excessive contraction that harms shape readability.
            k_pressure_eff = self.k_pressure * (0.65 + 0.35 * size_affine)
            scale_min_eff = max(self.scale_min, 1.00)
        else:
            k_pressure_eff = self.k_pressure * size_affine
            scale_min_eff = self.scale_min
        # If effective distance is smaller than safety threshold, scale up.
        raw = self.min_dist / effective
        return float(np.clip(1.0 + k_pressure_eff * (raw - 1.0), scale_min_eff, self.scale_max))

    def initiate_safe_return(self, start_poses, home_poses):
        self.is_returning = True
        n = min(len(start_poses), len(home_poses))
        self.return_start_poses = start_poses[:n].copy()
        self.return_home_poses = home_poses[:n].copy()

        self.moving_mask = np.zeros(n, dtype=bool)
        m = min(self.current_active_num, n)
        if m > 0:
            self.moving_mask[:m] = True
            max_dist = np.max(np.linalg.norm(self.return_home_poses[:m] - self.return_start_poses[:m], axis=1))
        else:
            max_dist = 0.0

        self.return_duration = max(max_dist / (self.max_vel * 0.45), 6.0)
        self.return_start_time = time.time()
        self.goals = self.return_start_poses.copy()

    def distribute_goals(self, start, goals, shape_num=None, active_num=None):
        if shape_num is None:
            shape_num = len(goals)
        if active_num is None:
            active_num = len(goals)

        active_num = min(active_num, len(start), len(goals))
        shape_num = min(shape_num, active_num)

        self.current_shape_num = shape_num
        self.current_active_num = active_num

        out_goals = np.copy(goals)
        self.drone_states.fill(0)

        if active_num > 0:
            shape_start = start[:shape_num]
            shape_goals = goals[:shape_num]

            if self.enable_dca and shape_num > 1:
                shape_goals = shape_goals + np.random.normal(0, 1e-3, shape_goals.shape)
                dist_matrix_llm = cdist(shape_goals, shape_goals)
                np.fill_diagonal(dist_matrix_llm, np.inf)

                effective_min = max(float(np.median(np.min(dist_matrix_llm, axis=1))), 0.02)
                target_spacing = self.min_dist * 1.05
                scale = np.clip(target_spacing / effective_min, 0.45, 3.5)

                centroid = np.mean(shape_goals, axis=0)
                scaled_shape_goals = centroid + (shape_goals - centroid) * scale

                for _ in range(50):
                    d = cdist(scaled_shape_goals, scaled_shape_goals)
                    np.fill_diagonal(d, np.inf)
                    if np.min(np.min(d, axis=1)) >= target_spacing * 0.99:
                        break
                    disp = np.zeros_like(scaled_shape_goals)
                    for i in range(len(scaled_shape_goals)):
                        mask = d[i] < target_spacing
                        if np.any(mask):
                            vec = scaled_shape_goals[i] - scaled_shape_goals[mask]
                            ds = d[i][mask].reshape(-1, 1)
                            ds_safe = np.maximum(ds, 1e-4)
                            pushes = (vec / ds_safe) * (target_spacing - ds) * 0.22
                            disp[i] = np.sum(pushes, axis=0)
                    scaled_shape_goals += np.clip(disp, -0.08, 0.08)

                cost = cdist(shape_start, scaled_shape_goals) ** 2.0
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    out_goals[r] = scaled_shape_goals[c]
                    self.drone_states[r] = 1

                for i in range(shape_num, active_num):
                    out_goals[i] = goals[i]
                    self.drone_states[i] = 2
            else:
                if shape_num > 0:
                    cost = cdist(shape_start, shape_goals)
                    for i in range(shape_num):
                        idx = np.argmin(cost[i])
                        out_goals[i] = shape_goals[idx]
                        self.drone_states[i] = 1
                        cost[i, :] = np.inf
                        cost[:, idx] = np.inf
                for i in range(shape_num, active_num):
                    out_goals[i] = goals[i]
                    self.drone_states[i] = 2

        self.goals = out_goals
        self.nominal_goals = np.copy(out_goals)
        self.stress_weights = self._build_stress_weights(self.nominal_goals[:self.current_shape_num])
        self.affine_matrix = np.eye(3)

    def _compute_affine_goals(self, poses, active_n):
        if self.nominal_goals is None:
            return self.goals

        shape_n = min(self.current_shape_num, active_n)
        if shape_n <= 1:
            return self.goals

        g_nom = self.nominal_goals[:shape_n]
        p_cur = poses[:shape_n]

        cg = np.mean(g_nom, axis=0)
        cp = np.mean(p_cur, axis=0)
        g0 = g_nom - cg
        p0 = p_cur - cp

        # Least-squares affine fit from nominal to current shape.
        a_fit, _, _, _ = np.linalg.lstsq(g0, p0, rcond=None)
        a_fit = a_fit.T

        p_scale = self._estimate_pressure_scale(poses, shape_n)

        size_scale = float(np.clip(np.sqrt(24.0 / max(shape_n, 1)), 0.60, 1.00))
        size_affine = size_scale * size_scale * size_scale
        fit_blend_eff = self.fit_blend * size_affine

        # Keep nominal shape as anchor; only a small portion of fitted affine drift is allowed.
        a_struct = (1.0 - fit_blend_eff) * np.eye(3) + fit_blend_eff * a_fit
        a_struct = self._clip_affine_matrix(a_struct)

        z_scale = 1.0 + self.z_scale_gain * (p_scale - 1.0)
        z_scale = float(np.clip(z_scale, 0.92, 1.25))
        a_pressure = np.diag([p_scale, p_scale, z_scale])
        a_target = a_pressure @ a_struct

        self.affine_matrix = self.affine_smooth * self.affine_matrix + (1.0 - self.affine_smooth) * a_target

        aff_goals = np.copy(self.goals)

        # AFC on XY plane around nominal centroid (with light blend to current centroid).
        center_xy = (1.0 - self.center_blend) * cg[:2] + self.center_blend * cp[:2]
        transformed = cg + (g0 @ self.affine_matrix.T)
        transformed[:, 0] += (center_xy[0] - cg[0])
        transformed[:, 1] += (center_xy[1] - cg[1])
        aff_goals[:shape_n, :2] = transformed[:, :2]

        # Preserve nominal 3D height profile: avoid pulling Z back to ground at takeoff.
        z_center = (1.0 - self.z_center_blend) * cg[2] + self.z_center_blend * cp[2]
        aff_goals[:shape_n, 2] = z_center + (g_nom[:, 2] - cg[2])
        return aff_goals

    def get_control(self, poses) -> None:
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

        self.goals = np.nan_to_num(self.goals)
        active_n = min(self.current_active_num, n)
        shape_n_global = min(self.current_shape_num, active_n)
        size_scale_global = float(np.clip(np.sqrt(24.0 / max(shape_n_global, 1)), 0.60, 1.00))
        if shape_n_global <= 30:
            sep_gain = 1.10
            safety_dist = self.min_dist * 1.05
        elif shape_n_global >= 60:
            sep_gain = 1.12
            safety_dist = self.min_dist * 1.06
        else:
            # Smooth transition between small and large swarm regimes.
            t = float(np.clip((shape_n_global - 30.0) / 30.0, 0.0, 1.0))
            sep_gain = (1.0 - t) * 1.10 + t * 1.12
            safety_dist = self.min_dist * ((1.0 - t) * 1.05 + t * 1.06)
        target_goals = self._compute_affine_goals(poses, active_n)

        ball_tree = BallTree(poses, metric="euclidean")
        control_vels = np.zeros_like(poses)

        for i, pose in enumerate(poses):
            if i >= active_n:
                continue

            shape_n = min(self.current_shape_num, active_n)
            size_scale = float(np.clip(np.sqrt(24.0 / max(shape_n, 1)), 0.60, 1.00))
            size_affine = size_scale * size_scale * size_scale
            k_affine_eff = self.k_affine * size_affine
            anchor_eff = float(np.clip(self.nominal_anchor_gain + 0.26 * (1.0 - size_scale), 0.0, 0.99))

            if self.nominal_goals is not None and i < len(self.nominal_goals):
                goal_nom = self.nominal_goals[i]
                to_goal_nom = goal_nom - pose
            else:
                goal_nom = target_goals[i]
                to_goal_nom = np.zeros(3)

            # AFC correction fades near terminal nominal goal to avoid residual bias.
            dist_nom = np.linalg.norm(to_goal_nom)
            aff_weight = np.clip(dist_nom / max(self.affine_fade_radius, 1e-3), 0.0, 1.0)
            aff_weight *= size_affine
            goal_used = goal_nom + aff_weight * (target_goals[i] - goal_nom)
            to_goal = goal_used - pose

            to_goal_mix = (1.0 - anchor_eff) * to_goal + anchor_eff * to_goal_nom
            dist_goal = np.linalg.norm(to_goal_mix)
            scale = dist_goal / 0.02 if dist_goal < 0.02 else 1.0
            v_nom = self.p_cohesion * to_goal_mix * scale
            v_nom[2] *= self.z_vel_boost
            if np.linalg.norm(v_nom) > self.max_vel:
                v_nom = (v_nom / np.linalg.norm(v_nom)) * self.max_vel

            nearest_ind = ball_tree.query_radius(pose.reshape(1, -1), safety_dist * 2.0)[0][1:]
            v_rep = np.zeros(3)
            local_neighbors = len(nearest_ind)
            for j in nearest_ind:
                rel = pose - poses[j]
                d = np.linalg.norm(rel)
                if d < safety_dist:
                    d_safe = max(d, 0.01)
                    strength = self.p_separation * sep_gain
                    if shape_n_global >= 60:
                        # Dense large swarms get extra safety margin only when local crowding is high.
                        crowd_boost = 1.0 + 0.015 * np.clip(local_neighbors - 4, 0, 8)
                        strength *= crowd_boost
                    penetration = safety_dist / d_safe
                    if shape_n_global <= 30:
                        emergency_ratio = 0.88
                        emergency_mult = 1.03
                    elif shape_n_global >= 60:
                        emergency_ratio = 0.905
                        emergency_mult = 1.06
                    else:
                        t = float(np.clip((shape_n_global - 30.0) / 30.0, 0.0, 1.0))
                        emergency_ratio = (1.0 - t) * 0.88 + t * 0.905
                        emergency_mult = (1.0 - t) * 1.03 + t * 1.06
                    if d < emergency_ratio * safety_dist:
                        strength *= emergency_mult
                    dyn = np.clip(1.2 * (penetration ** 20), 1.2, 3.0)
                    strength *= dyn
                    rep_mag = strength * (1.0 / d_safe - 1.0 / safety_dist) / (d_safe ** 2 + 0.01)
                    v_rep += rep_mag * (rel / d_safe)

                    # Add a light buffer near min_dist to suppress boundary-crossing jitter.
                    stab_dist = self.min_dist * self.min_dist_stab_ratio
                    if d_safe < stab_dist:
                        margin = (stab_dist - d_safe) / max(stab_dist, 1e-3)
                        v_rep += self.min_dist_stab_gain * margin * (rel / d_safe)

            v_aff = np.zeros(3)
            if self.stress_weights is not None and i < shape_n:
                for j in range(shape_n):
                    if i == j:
                        continue
                    wij = self.stress_weights[i, j]
                    if wij <= 1e-6:
                        continue
                    desired_ij = target_goals[i] - target_goals[j]
                    err_ij = (poses[i] - poses[j]) - desired_ij
                    v_aff += -k_affine_eff * wij * err_ij

            aff_norm = np.linalg.norm(v_aff)
            aff_cap = self.max_vel * self.max_affine_correction_ratio
            if aff_norm > aff_cap:
                v_aff = (v_aff / aff_norm) * aff_cap

            v_aff *= aff_weight

            v_cmd = v_nom + v_rep + v_aff

            # Near-goal velocity scheduling reduces residual jitter in dense formations.
            near_scale = np.clip(dist_goal / max(self.near_goal_radius, 1e-3), self.near_goal_min_ratio, 1.0)
            local_max_vel = max(self.max_vel * self.near_goal_min_ratio, self.max_vel * near_scale)
            speed = np.linalg.norm(v_cmd)
            if speed > local_max_vel:
                v_cmd = (v_cmd / speed) * local_max_vel

            control_vels[i] = v_cmd

        if self.is_returning:
            mask = getattr(self, "moving_mask", np.ones(n, dtype=bool))
            for i in range(n):
                if not mask[i]:
                    control_vels[i] = 0.0
        else:
            for i in range(n):
                if i >= active_n:
                    control_vels[i] = 0.0

        control_vels = 0.80 * control_vels + 0.20 * self.velocities[:n]

        if self.is_returning:
            mask = getattr(self, "moving_mask", np.ones(n, dtype=bool))
            for i in range(n):
                if not mask[i]:
                    control_vels[i] = 0.0
                    self.velocities[i] = 0.0
        else:
            for i in range(n):
                if i >= active_n:
                    control_vels[i] = 0.0
                    self.velocities[i] = 0.0

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
        generate_idle_report(self, home_poses)

    def generate_fms_srm_report(self, phase_name, phase_idx):
        generate_fms_srm_report(self, phase_name, phase_idx)
