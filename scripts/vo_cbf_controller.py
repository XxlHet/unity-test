#!/usr/bin/env python3

"""
VO + CBF controller (Python re-implementation inspired by KTH-RPL/MA-CBF-VO).

Design goals:
- Keep this file standalone and non-invasive to the existing codebase.
- Provide an API close to APFSwarmController so it can be plugged in for A/B tests.
- Combine:
  1) VO-like constraint as a soft penalty (via slack variables)
  2) CBF-style hard safety constraints for collision avoidance.

Notes:
- This is a practical re-implementation for your experiment stack (velocity command output),
  not a line-by-line port of the MATLAB Simulink system.
- In this project, `get_control` returns velocity commands (m/s), so we optimize accelerations
  and integrate them over one control timestep.
"""

import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist

try:
    from apf_data_collector import collect_step_data
    from apf_plotter import generate_fms_srm_report, generate_idle_report, generate_plots
except ImportError:
    from scripts.apf_data_collector import collect_step_data
    from scripts.apf_plotter import generate_fms_srm_report, generate_idle_report, generate_plots


class VOCBFSwarmController:
    def __init__(
        self,
        max_vel: float = 0.9,
        min_dist: float = 0.35,
        timestep: float = 0.05,
        p_gain: float = 1.0,
        d_gain: float = 0.5,
        max_acc: float = 1.2,
        alpha_c: float = 8.0,
        beta_c: float = 6.0,
        alpha_vo: float = 10.0,
        k_u: float = 1.0,
        k_vo: float = 300.0,
        geometric_tol: float = 0.08,
        consider_only_min_ttc: bool = False,
        include_vo_as_hard_constraint: bool = False,
        use_safe_cbf: bool = True,
    ) -> None:
        self.swarm = None
        self.goals = None

        self.min_dist = float(min_dist)
        self.max_vel = float(max_vel)
        self.max_acc = float(max_acc)
        self.timestep = float(timestep)

        self.p_gain = float(p_gain)
        self.d_gain = float(d_gain)

        self.alpha_c = float(alpha_c)
        self.beta_c = float(beta_c)
        self.alpha_vo = float(alpha_vo)
        self.k_u = float(k_u)
        self.k_vo = float(k_vo)
        self.geometric_tol = float(geometric_tol)

        self.consider_only_min_ttc = bool(consider_only_min_ttc)
        self.include_vo_as_hard_constraint = bool(include_vo_as_hard_constraint)
        self.use_safe_cbf = bool(use_safe_cbf)

        # Compatibility fields expected by your pipeline.
        self.enable_dca = False
        self.log_dir = ""
        self.current_log_name = ""
        self.csv_initialized = False
        self.start_time = 0.0
        self.last_csv_path = ""

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

        # Return/FLSM compatibility states.
        self.is_returning = False
        self.return_start_poses = None
        self.return_home_poses = None
        self.return_start_time = 0.0
        self.return_duration = 5.0
        self.return_home_tol = 0.08

        self.current_shape_num = 0
        self.current_active_num = 0
        self.moving_mask = None

        self.global_home_poses = None
        self.velocities = None
        self.prev_position_error = None

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

        self.return_duration = max(max_dist / max(self.max_vel * 0.45, 1e-3), 6.0)
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
            if shape_num > 0:
                cost_matrix = cdist(shape_start, shape_goals)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    out_goals[r] = shape_goals[c]
                    self.drone_states[r] = 1

            for i in range(shape_num, active_num):
                out_goals[i] = goals[i]
                self.drone_states[i] = 2

        self.goals = out_goals

    @staticmethod
    def _ttc_disc(p_i, p_j, v_i, v_j, radius_sum) -> float:
        # Same quadratic TTC idea used in the referenced MATLAB repo.
        err_p = p_i - p_j
        err_v = v_i - v_j

        a = float(np.dot(err_v, err_v))
        b = float(2.0 * np.dot(err_p, err_v))
        c = float(np.dot(err_p, err_p) - radius_sum * radius_sum)

        if c < 0:
            return 1e-3
        if a < 1e-9:
            return np.inf

        roots = np.roots([a, b, c])
        real_pos = [float(r.real) for r in roots if np.isreal(r) and r.real > 0]
        return min(real_pos) if real_pos else np.inf

    def _desired_accel(self, p: np.ndarray, v: np.ndarray, p_goal: np.ndarray, idx: int) -> np.ndarray:
        err = p_goal - p
        dist = np.linalg.norm(err)
        if dist > 1e-6:
            v_target = (err / dist) * self.max_vel
        else:
            v_target = np.zeros(3)

        if self.prev_position_error is None:
            self.prev_position_error = np.zeros_like(self.goals)

        d_err = (err - self.prev_position_error[idx]) / max(self.timestep, 1e-3)
        self.prev_position_error[idx] = err

        u_des = self.p_gain * (v_target - v) + self.d_gain * d_err
        u_norm = np.linalg.norm(u_des)
        if u_norm > self.max_acc:
            u_des = u_des / u_norm * self.max_acc
        return u_des

    def _solve_agent_qp(
        self,
        p_i: np.ndarray,
        v_i: np.ndarray,
        u_des: np.ndarray,
        obs_states: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        # Decision variables: [u_x, u_y, u_z, lambda_0, ..., lambda_m-1]
        m = len(obs_states)
        nvar = 3 + m

        x0 = np.zeros(nvar)
        x0[:3] = u_des

        bounds = [(-self.max_acc, self.max_acc)] * 3
        bounds += [(0.0, None)] * m

        # Precompute TTC-based weights for VO soft penalties.
        ttc_vec = []
        for p_j, v_j in obs_states:
            ttc = self._ttc_disc(p_i, p_j, v_i, v_j, self.min_dist * (1.0 + self.geometric_tol))
            ttc_vec.append(ttc)
        ttc_vec = np.array(ttc_vec, dtype=float) if m > 0 else np.array([], dtype=float)

        if m > 0:
            if self.consider_only_min_ttc:
                idx_min = int(np.argmin(ttc_vec)) if np.isfinite(ttc_vec).any() else 0
                w = np.zeros(m)
                w[idx_min] = 1.0 / max(ttc_vec[idx_min], 1e-3)
            else:
                w = np.array([0.0 if np.isinf(t) else 1.0 / max(t, 1e-3) for t in ttc_vec])
        else:
            w = np.array([], dtype=float)

        def objective(x):
            u = x[:3]
            base = self.k_u * float(np.dot(u - u_des, u - u_des))
            if m == 0 or self.include_vo_as_hard_constraint:
                return base
            lam = x[3:]
            return base + self.k_vo * float(np.dot(w * lam, lam))

        constraints = []

        # Per-obstacle CBF hard constraints + VO constraints.
        for j, (p_j, v_j) in enumerate(obs_states):
            p_rel = p_i - p_j
            v_rel = v_i - v_j
            r_eff = self.min_dist * (1.0 + self.geometric_tol)

            if self.use_safe_cbf:
                # Relative degree-2 CBF for double integrator:
                # h = ||p_rel||^2 - r^2
                # enforce h_ddot + alpha_c * h_dot + beta_c * h >= 0
                # with h_dot = 2 p_rel·v_rel
                # and h_ddot = 2(v_rel·v_rel + p_rel·u)
                h = float(np.dot(p_rel, p_rel) - r_eff * r_eff)
                h_dot = float(2.0 * np.dot(p_rel, v_rel))

                def cbf_fun(x, p_rel=p_rel, v_rel=v_rel, h=h, h_dot=h_dot):
                    u = x[:3]
                    h_ddot = 2.0 * (np.dot(v_rel, v_rel) + np.dot(p_rel, u))
                    return h_ddot + self.alpha_c * h_dot + self.beta_c * h

                # Include safe CBF only when agents are approaching each other.
                # This mirrors the "closing speed" check in the MATLAB code.
                if np.dot(v_rel, p_rel) < 0:
                    constraints.append({"type": "ineq", "fun": cbf_fun})

            # VO-like constraint evaluated on next relative velocity v_rel_next = v_rel + dt*u.
            # Keep it hard or softened via slack lambda_j.
            def vo_margin(x, p_rel=p_rel, v_rel=v_rel):
                u = x[:3]
                v_next = v_rel + self.timestep * u
                return float(np.dot(p_rel, v_next) + np.linalg.norm(p_rel) * np.linalg.norm(v_next))

            if self.include_vo_as_hard_constraint:
                constraints.append({"type": "ineq", "fun": vo_margin})
            else:
                def vo_soft(x, j=j):
                    lam_j = x[3 + j]
                    return vo_margin(x) + lam_j

                constraints.append({"type": "ineq", "fun": vo_soft})

        try:
            sol = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 80, "ftol": 1e-5, "disp": False},
            )
            if (not sol.success) or np.any(~np.isfinite(sol.x[:3])):
                return np.clip(u_des, -self.max_acc, self.max_acc)
            return np.clip(sol.x[:3], -self.max_acc, self.max_acc)
        except Exception:
            # Conservative fallback to keep runtime robust.
            return np.clip(u_des, -self.max_acc, self.max_acc)

    def get_control(self, poses):
        step_start_time = time.time()

        if self.goals is None:
            return np.zeros_like(poses)

        n = min(self.goals.shape[0], poses.shape[0])
        poses = poses[:n]

        if self.velocities is None:
            self.velocities = np.zeros_like(poses)
        if self.prev_position_error is None:
            self.prev_position_error = np.zeros_like(poses)

        # Return trajectory shaping (kept for compatibility with FLSM flow).
        if self.is_returning and self.return_home_poses is not None and self.return_start_poses is not None:
            elapsed = time.time() - self.return_start_time
            progress = min(elapsed / max(self.return_duration, 1e-3), 1.0)
            smooth_p = progress * progress * (3.0 - 2.0 * progress)

            m = min(self.current_active_num, n)
            if m > 0:
                center = np.mean(self.return_home_poses[:m], axis=0)
                bloom_scale = 1.0 + 1.1 * np.sin(np.pi * smooth_p)
                bloomed = center + (self.return_home_poses[:m] - center) * bloom_scale
                self.goals[:m] = self.return_start_poses[:m] + (bloomed - self.return_start_poses[:m]) * smooth_p

        control_vels = np.zeros_like(poses)
        self.goals = np.nan_to_num(self.goals)

        # Only active drones are controlled.
        active_n = self.current_active_num if not self.is_returning else min(self.current_active_num, n)
        active_n = min(active_n, n)

        for i in range(active_n):
            p_i = poses[i]
            v_i = self.velocities[i]
            g_i = self.goals[i]
            u_des = self._desired_accel(p_i, v_i, g_i, i)

            obs_states = []
            for j in range(active_n):
                if j == i:
                    continue
                obs_states.append((poses[j], self.velocities[j]))

            u_opt = self._solve_agent_qp(p_i, v_i, u_des, obs_states)
            v_cmd = v_i + self.timestep * u_opt

            speed = np.linalg.norm(v_cmd)
            if speed > self.max_vel:
                v_cmd = (v_cmd / speed) * self.max_vel

            control_vels[i] = v_cmd

        # Non-active drones stay still.
        for i in range(active_n, n):
            control_vels[i] = 0.0
            self.velocities[i] = 0.0

        # Return mode lock near home.
        if self.is_returning and self.return_home_poses is not None:
            m = min(active_n, len(self.return_home_poses))
            for i in range(m):
                if np.linalg.norm(poses[i] - self.return_home_poses[i]) < self.return_home_tol:
                    self.goals[i] = self.return_home_poses[i]
                    control_vels[i] = 0.0
                    self.velocities[i] = 0.0

        self.velocities[:n] = control_vels.copy()
        collect_step_data(self, poses, control_vels, n, step_start_time)
        return control_vels

    def generate_plots(self):
        generate_plots(self)

    def generate_idle_report(self, home_poses):
        generate_idle_report(self, home_poses)

    def generate_fms_srm_report(self, phase_name, phase_idx):
        generate_fms_srm_report(self, phase_name, phase_idx)
