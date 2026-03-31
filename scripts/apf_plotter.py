#!/usr/bin/env python3

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _prepare_plot_series(df, col):
    if "Time(s)" not in df.columns or col not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    plot_df = df[["Time(s)", col]].copy()
    plot_df["Time(s)"] = pd.to_numeric(plot_df["Time(s)"], errors="coerce")
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["Time(s)", col])
    if plot_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    return plot_df["Time(s)"], plot_df[col]


def generate_plots(controller):
    if not controller.last_csv_path or not os.path.exists(controller.last_csv_path):
        return

    mode_prefix = "DCA" if controller.enable_dca else "Base"
    algo_label = "DCA (Ours)" if controller.enable_dca else "Baseline"

    print(f"\n[*] Generating plots for [{algo_label}] mode...")
    try:
        df = pd.read_csv(controller.last_csv_path)
        if df.empty:
            print(f"⚠️ Plotting skipped: no metric rows in {controller.last_csv_path}")
            return
        metrics = {
            "Target_Error(m)": (
                "Convergence Error Comparison",
                "Mean Error (m)",
                "#2ECC71" if controller.enable_dca else "#E74C3C",
            ),
            "Min_Distance(m)": (
                "Minimum Distance Comparison",
                "Min Distance (m)",
                "#2ECC71" if controller.enable_dca else "#E74C3C",
            ),
            "Avg_Velocity(m/s)": (
                "Average Velocity Comparison",
                "Avg Velocity (m/s)",
                "#2ECC71" if controller.enable_dca else "#E74C3C",
            ),
            "Comp_Time(ms)": (
                "Computation Time per Step",
                "Time (ms)",
                "#F39C12" if controller.enable_dca else "#8E44AD",
            ),
            "Collisions": (
                "Cumulative Safety Violations",
                "Violation Count",
                "#E74C3C",
            ),
        }
        for col, (title, ylabel, color) in metrics.items():
            if col in df.columns:
                x_data, y_data = _prepare_plot_series(df, col)
                if x_data.empty or y_data.empty:
                    print(f"⚠️ Plot skipped for '{col}': no valid numeric rows.")
                    continue

                plt.figure(figsize=(9, 5.5))

                if col == "Collisions":
                    collisions_cum = y_data.cumsum()
                    plt.plot(x_data, collisions_cum, linewidth=2.5, color=color, label=algo_label)
                else:
                    plt.plot(
                        x_data,
                        y_data,
                        linewidth=2.5 if controller.enable_dca else 1.5,
                        color=color,
                        linestyle="-" if controller.enable_dca else "--",
                        label=algo_label,
                        alpha=0.9,
                    )

                if col == "Target_Error(m)":
                    plt.axhline(y=0.0, color="black", linestyle=":", label="Ideal")
                elif col == "Min_Distance(m)":
                    plt.axhline(y=controller.min_dist, color="black", linestyle="-.", label=f"Safety Limit ({controller.min_dist}m)")
                    plt.axhspan(0, controller.min_dist, color="gray", alpha=0.15)
                    plt.axhline(y=0.3, color="red", linestyle="--", alpha=0.6, label="Physical Collision (0.3m)")
                    plt.ylim(bottom=max(0, controller.min_dist - 0.05), top=float(y_data.max()) * 1.05)
                elif col == "Avg_Velocity(m/s)":
                    plt.axhline(y=controller.max_vel, color="blue", linestyle=":", alpha=0.5, label="Max Velocity")
                    plt.ylim(bottom=-0.05, top=controller.max_vel + 0.1)
                elif col == "Comp_Time(ms)":
                    plt.axhline(y=10.0, color="red", linestyle=":", label="100Hz Deadline (10ms)")
                    plt.ylim(bottom=0.0, top=max(15.0, float(y_data.max()) * 1.2))
                elif col == "Collisions":
                    plt.axhline(y=0, color="black", linestyle=":", label="Ideal (Zero)")
                    plt.ylim(bottom=-0.5, top=max(1, float(collisions_cum.max()) + 1.5))

                plt.title(title, fontweight="bold", fontsize=14)
                plt.xlabel("Time $t$ (s)", fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.legend(loc="best", fontsize=11, frameon=True, shadow=True)

                img_name = f"{mode_prefix}_{controller.current_log_name}_{col.split('(')[0]}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(controller.log_dir, img_name), dpi=300)
                plt.close()
        print(f"[*] Plots saved: {controller.log_dir}")
    except Exception as e:
        print(f"⚠️ Plotting Error: {e}")


def generate_idle_report(controller, home_poses):
    if not getattr(controller, "fms_dir", ""):
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Phase 0: FLSM Fleet Standby", fontweight="bold")
    ax.scatter(home_poses[:, 0], home_poses[:, 1], home_poses[:, 2], color="gray", marker="o", s=30, label="Idle Fleet")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(controller.fms_dir, "00_FLSM_Standby.png"), dpi=300)
    plt.close()


def _safe_id_array(ids, upper_bound):
    arr = np.asarray(ids, dtype=int).reshape(-1)
    return arr[(arr >= 0) & (arr < upper_bound)]


def _draw_3d_group(ax, df, ids, color, linestyle, linewidth, alpha, label, max_draw):
    if ids.size == 0:
        return

    existing_ids = sorted(set(df["DroneID"].astype(int).tolist()).intersection(set(ids.tolist())))
    if not existing_ids:
        return

    if len(existing_ids) > max_draw:
        pos = np.linspace(0, len(existing_ids) - 1, max_draw, dtype=int)
        draw_ids = [existing_ids[p] for p in pos]
    else:
        draw_ids = existing_ids

    first = True
    for drone_id in draw_ids:
        traj = df[df["DroneID"] == drone_id]
        if len(traj) < 2:
            continue
        ax.plot(
            traj["X"],
            traj["Y"],
            traj["Z"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if first else "",
        )
        first = False


def _draw_2d_group(ax, df, ids, color, linestyle, linewidth, alpha, label, max_draw):
    if ids.size == 0:
        return

    existing_ids = sorted(set(df["DroneID"].astype(int).tolist()).intersection(set(ids.tolist())))
    if not existing_ids:
        return

    if len(existing_ids) > max_draw:
        pos = np.linspace(0, len(existing_ids) - 1, max_draw, dtype=int)
        draw_ids = [existing_ids[p] for p in pos]
    else:
        draw_ids = existing_ids

    first = True
    for drone_id in draw_ids:
        traj = df[df["DroneID"] == drone_id]
        if len(traj) < 2:
            continue
        ax.plot(
            traj["X"],
            traj["Y"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if first else "",
        )
        first = False


def generate_fms_srm_report(controller, phase_name, phase_idx):
    if not controller.trajectory_log or not getattr(controller, "fms_dir", ""):
        return

    print(f"\n[*] 📊 [FLSM] Generating Trajectory Map for Phase {phase_idx}...")
    df = pd.DataFrame(controller.trajectory_log, columns=["Time", "DroneID", "X", "Y", "Z", "State", "MinDist"])

    drone_ids = sorted(df["DroneID"].astype(int).unique().tolist())
    drone_count = len(drone_ids)
    max_draw = 72 if drone_count <= 84 else 96

    phase_prev = int(getattr(controller, "phase_prev_active_num", 0))
    phase_shape = int(getattr(controller, "phase_shape_num", 0))
    phase_active = int(getattr(controller, "phase_active_num", phase_shape))
    phase_prev = max(0, min(phase_prev, drone_count))
    phase_shape = max(0, min(phase_shape, drone_count))
    phase_active = max(0, min(phase_active, drone_count))

    new_launch_ids = _safe_id_array(getattr(controller, "phase_new_launch_ids", np.array([], dtype=int)), drone_count)
    return_ids = _safe_id_array(getattr(controller, "phase_return_ids", np.array([], dtype=int)), drone_count)

    if new_launch_ids.size == 0 and phase_active > phase_prev:
        new_launch_ids = np.arange(phase_prev, phase_active, dtype=int)
    if return_ids.size == 0 and phase_prev > phase_shape:
        return_ids = np.arange(phase_shape, phase_prev, dtype=int)

    shape_ids = np.arange(phase_shape, dtype=int)
    stayed_shape_ids = np.arange(min(phase_prev, phase_shape), dtype=int)

    home = getattr(controller, "global_home_poses", None)

    fig = plt.figure(figsize=(24, 7.2))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title(f"Phase {phase_idx}: [{phase_name}] Trajectory Groups", fontweight="bold")

    if isinstance(home, np.ndarray) and home.size > 0:
        ax1.scatter(home[:, 0], home[:, 1], home[:, 2], color="#D5DBDB", marker="o", s=10, alpha=0.35, label="Ground Grid")

    _draw_3d_group(ax1, df, stayed_shape_ids, "#2CA02C", "-", 1.1, 0.35, "Shape (already airborne)", max_draw)
    _draw_3d_group(ax1, df, new_launch_ids, "#1F77B4", "-", 1.8, 0.72, "New Launch This Phase", max_draw)
    _draw_3d_group(ax1, df, return_ids, "#FF7F0E", "--", 1.6, 0.76, "Returning This Phase", max_draw)

    last_df = df.sort_values("Time").groupby("DroneID", as_index=False).tail(1)
    if shape_ids.size > 0:
        shape_end = last_df[last_df["DroneID"].isin(shape_ids.tolist())]
        if not shape_end.empty:
            ax1.scatter(shape_end["X"], shape_end["Y"], shape_end["Z"], color="#2ECC71", marker="*", s=55, alpha=0.95, label="Shape Endpoints")
    if return_ids.size > 0:
        ret_end = last_df[last_df["DroneID"].isin(return_ids.tolist())]
        if not ret_end.empty:
            ax1.scatter(ret_end["X"], ret_end["Y"], ret_end["Z"], color="#AAB7B8", marker="o", s=20, alpha=0.9, label="Returned/Landed Endpoints")

    ax1.view_init(elev=23, azim=42)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend(loc="best", fontsize=9)

    ax2 = fig.add_subplot(132)
    ax2.set_title(f"Phase {phase_idx}: Ground Markers + Top View", fontweight="bold")
    if isinstance(home, np.ndarray) and home.size > 0:
        ax2.scatter(home[:, 0], home[:, 1], color="#D5DBDB", marker="o", s=10, alpha=0.30, label="Ground Grid")

    _draw_2d_group(ax2, df, new_launch_ids, "#1F77B4", "-", 1.4, 0.68, "Launch Trajectory", max_draw)
    _draw_2d_group(ax2, df, return_ids, "#FF7F0E", "--", 1.4, 0.72, "Return Trajectory", max_draw)

    if isinstance(home, np.ndarray) and home.size > 0:
        if new_launch_ids.size > 0:
            src = home[new_launch_ids]
            ax2.scatter(src[:, 0], src[:, 1], marker="^", s=34, color="#1F77B4", edgecolors="white", linewidths=0.5, label="New Launch Pads")
        if return_ids.size > 0:
            dst = home[return_ids]
            ax2.scatter(dst[:, 0], dst[:, 1], marker="s", s=26, facecolors="none", edgecolors="#FF7F0E", linewidths=1.0, label="Return Pads")

    if shape_ids.size > 0:
        shape_end_2d = last_df[last_df["DroneID"].isin(shape_ids.tolist())]
        if not shape_end_2d.empty:
            ax2.scatter(shape_end_2d["X"], shape_end_2d["Y"], marker="*", s=42, color="#2ECC71", alpha=0.95, label="Shape End")

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=9)

    ax3 = fig.add_subplot(133)
    ax3.set_title(f"Phase {phase_idx}: Min Distance / Collision Risk", fontweight="bold")
    time_dist = df.groupby("Time")["MinDist"].min().reset_index()
    time_dist = time_dist.sort_values("Time")
    ax3.plot(time_dist["Time"], time_dist["MinDist"], color="#2E86C1", linewidth=2.2, label="Min Inter-Drone Dist")

    hard_th = 0.30
    safe_th = float(getattr(controller, "min_dist", 0.35))
    ax3.axhline(y=safe_th, color="#566573", linestyle="-.", linewidth=1.8, label=f"Safety Baseline ({safe_th:.2f}m)")
    ax3.axhline(y=hard_th, color="#C0392B", linestyle="--", linewidth=1.8, label=f"Hard Collision ({hard_th:.2f}m)")
    ax3.fill_between(time_dist["Time"], 0, safe_th, color="#F5B7B1", alpha=0.20)

    hard_hits = time_dist[time_dist["MinDist"] < hard_th]
    if not hard_hits.empty:
        ax3.scatter(hard_hits["Time"], hard_hits["MinDist"], color="#C0392B", marker="x", s=20, label="Hard Collision Events")

    violation_steps = int((time_dist["MinDist"] < safe_th).sum())
    hard_steps = int((time_dist["MinDist"] < hard_th).sum())
    ax3.text(
        0.02,
        0.98,
        f"Violations: {violation_steps}  |  Hard: {hard_steps}",
        transform=ax3.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#CCD1D1"},
    )

    ax3.set_ylim(bottom=0.0)
    ax3.set_xlabel("Phase Time (s)")
    ax3.set_ylabel("Distance (m)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Phase {phase_idx}: prev_active={phase_prev}, target_shape={phase_shape}, new_launch={len(new_launch_ids)}, return={len(return_ids)}",
        fontsize=11,
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(controller.fms_dir, f"{phase_idx:02d}_{phase_name}.png"), dpi=320)
    plt.close()
