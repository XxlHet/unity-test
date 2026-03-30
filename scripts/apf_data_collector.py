#!/usr/bin/env python3

import csv
import os
import time

import numpy as np


def collect_step_data(controller, poses, control_vels, n, step_start_time):
    _collect_trajectory_sample(controller, poses, n)
    _append_metrics_row(controller, poses, control_vels, step_start_time)


def _collect_trajectory_sample(controller, poses, n):
    controller.frame_counter += 1
    if controller.frame_counter % 10 != 0:
        return
    if not getattr(controller, "fms_dir", ""):
        return

    current_t = time.time() - controller.phase_start_time

    active_indices = np.asarray(getattr(controller, "active_indices", np.arange(controller.current_active_num)), dtype=int)
    active_indices = active_indices[(active_indices >= 0) & (active_indices < len(poses))]

    active_poses = poses[active_indices] if active_indices.size > 0 else np.zeros((0, 3))
    if active_indices.size > 1:
        dists = np.linalg.norm(active_poses[:, np.newaxis, :] - active_poses[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        phase_min_dist = np.min(dists)
    else:
        phase_min_dist = controller.min_dist

    moving_mask = getattr(controller, "moving_mask", np.ones(n, dtype=bool))
    shape_indices = np.asarray(getattr(controller, "shape_assignment_indices", np.array([], dtype=int)), dtype=int)
    shape_set = set(shape_indices.tolist())

    for i in range(len(poses)):
        state = 0
        if controller.is_returning:
            if moving_mask[i]:
                state = 2
        elif i in shape_set:
            state = 1
        elif i in active_indices:
            state = 2

        if state != 0 or poses[i][2] > 0.1:
            controller.trajectory_log.append([
                current_t,
                i,
                poses[i][0],
                poses[i][1],
                poses[i][2],
                state,
                phase_min_dist,
            ])


def _append_metrics_row(controller, poses, control_vels, step_start_time):
    if not controller.log_dir or not controller.current_log_name or controller.current_shape_num <= 0:
        return

    full_path = os.path.join(controller.log_dir, f"{controller.current_log_name}.csv")
    if controller.last_csv_path != full_path:
        controller.csv_initialized = False
        controller.last_csv_path = full_path

    if not controller.csv_initialized:
        try:
            with open(full_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Time(s)",
                    "Min_Distance(m)",
                    "Avg_Velocity(m/s)",
                    "Target_Error(m)",
                    "Comp_Time(ms)",
                    "Collisions",
                ])
            controller.start_time = time.time()
            controller.csv_initialized = True
        except Exception:
            return

    curr_t = round(time.time() - controller.start_time, 2)
    shape_indices = np.asarray(getattr(controller, "shape_assignment_indices", np.array([], dtype=int)), dtype=int)
    shape_indices = shape_indices[(shape_indices >= 0) & (shape_indices < len(poses))]
    if shape_indices.size == 0:
        return

    eval_poses = poses[shape_indices]
    eval_goals = controller.goals[shape_indices]
    eval_vels = control_vels[shape_indices]

    if shape_indices.size > 1:
        diffs = eval_poses[:, np.newaxis, :] - eval_poses[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        finite_dists = dists[np.isfinite(dists)]
        min_d = round(float(np.min(finite_dists)), 4) if finite_dists.size else 0.0
    else:
        min_d = 0.0

    vel_norm = np.linalg.norm(eval_vels, axis=1)
    vel_norm = vel_norm[np.isfinite(vel_norm)]
    avg_v = round(float(np.mean(vel_norm)), 4) if vel_norm.size else 0.0

    goal_err = np.linalg.norm(eval_goals - eval_poses, axis=1)
    goal_err = goal_err[np.isfinite(goal_err)]
    err = round(float(np.mean(goal_err)), 4) if goal_err.size else 0.0

    step_comp_time_ms = (time.time() - step_start_time) * 1000.0
    current_collisions = 1 if (0 < min_d < 0.3) else 0

    try:
        with open(full_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([curr_t, min_d, avg_v, err, round(step_comp_time_ms, 2), current_collisions])
    except Exception:
        pass
