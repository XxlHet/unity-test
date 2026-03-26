#!/usr/bin/env python3

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(controller):
    if not controller.last_csv_path or not os.path.exists(controller.last_csv_path):
        return

    mode_prefix = "DCA" if controller.enable_dca else "Base"
    algo_label = "DCA (Ours)" if controller.enable_dca else "Baseline"

    print(f"\n[*] Generating plots for [{algo_label}] mode...")
    try:
        df = pd.read_csv(controller.last_csv_path)
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
                "Cumulative Collisions",
                "Total Collisions",
                "#E74C3C",
            ),
        }
        for col, (title, ylabel, color) in metrics.items():
            if col in df.columns:
                plt.figure(figsize=(9, 5.5))

                if col == "Collisions":
                    plt.plot(df["Time(s)"], df[col].cumsum(), linewidth=2.5, color=color, label=algo_label)
                else:
                    plt.plot(
                        df["Time(s)"],
                        df[col],
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
                    plt.ylim(bottom=max(0, controller.min_dist - 0.05), top=df[col].max() * 1.05)
                elif col == "Avg_Velocity(m/s)":
                    plt.axhline(y=controller.max_vel, color="blue", linestyle=":", alpha=0.5, label="Max Velocity")
                    plt.ylim(bottom=-0.05, top=controller.max_vel + 0.1)
                elif col == "Comp_Time(ms)":
                    plt.axhline(y=10.0, color="red", linestyle=":", label="100Hz Deadline (10ms)")
                    plt.ylim(bottom=0.0, top=max(15.0, df[col].max() * 1.2))
                elif col == "Collisions":
                    plt.axhline(y=0, color="black", linestyle=":", label="Ideal (Zero)")
                    plt.ylim(bottom=-0.5, top=max(1, df[col].cumsum().max() + 1.5))

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
    ax.set_title("Phase 0: FMS Fleet Standby", fontweight="bold")
    ax.scatter(home_poses[:, 0], home_poses[:, 1], home_poses[:, 2], color="gray", marker="o", s=30, label="Idle Fleet")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(controller.fms_dir, "00_FMS_Standby.png"), dpi=300)
    plt.close()


def generate_fms_srm_report(controller, phase_name, phase_idx):
    if not controller.trajectory_log or not getattr(controller, "fms_dir", ""):
        return

    print(f"\n[*] 📊 [FMS] Generating Trajectory Map for Phase {phase_idx}...")
    df = pd.DataFrame(controller.trajectory_log, columns=["Time", "DroneID", "X", "Y", "Z", "State", "MinDist"])

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title(f"Phase {phase_idx}: [{phase_name}] Trajectory & FMS Status", fontweight="bold")

    if hasattr(controller, "global_home_poses") and controller.global_home_poses is not None:
        ax1.scatter(
            controller.global_home_poses[:, 0],
            controller.global_home_poses[:, 1],
            controller.global_home_poses[:, 2],
            color="lightgray",
            marker="o",
            s=20,
            alpha=0.5,
            label="Base Grid",
        )

    plotted = set()
    for drone_id in df["DroneID"].unique():
        drone_data = df[df["DroneID"] == drone_id]
        states = drone_data["State"].unique()

        if 1 in states:
            line = drone_data[drone_data["State"] == 1]
            lbl1 = "FMS Dispatch" if "Disp" not in plotted else ""
            ax1.plot(line["X"], line["Y"], line["Z"], color="#3498DB", alpha=0.2, linewidth=0.8, label=lbl1)

            end_pt = line.iloc[-1]
            lbl_star = "Shape Node" if "Star" not in plotted else ""
            ax1.scatter(
                end_pt["X"],
                end_pt["Y"],
                end_pt["Z"],
                color="#2ECC71",
                edgecolor="#196F3D",
                linewidth=0.5,
                marker="*",
                s=80,
                zorder=5,
                label=lbl_star,
            )
            plotted.update(["Disp", "Star"])

        if 2 in states:
            line = drone_data[drone_data["State"] == 2]
            lbl2 = "SRM Return" if "Ret" not in plotted else ""
            ax1.plot(
                line["X"],
                line["Y"],
                line["Z"],
                color="#E74C3C",
                alpha=0.25,
                linestyle=":",
                linewidth=1.0,
                label=lbl2,
            )

            end_pt = line.iloc[-1]
            lbl_dot = "Landed/Idle" if "Dot" not in plotted else ""
            ax1.scatter(
                end_pt["X"],
                end_pt["Y"],
                end_pt["Z"],
                color="lightgray",
                edgecolor="gray",
                linewidth=0.8,
                marker="o",
                s=35,
                zorder=5,
                label=lbl_dot,
            )
            plotted.update(["Ret", "Dot"])

    ax1.view_init(elev=25, azim=40)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_title(f"Phase {phase_idx}: Zero-Collision Proof", fontweight="bold")
    time_dist = df.groupby("Time")["MinDist"].min().reset_index()
    ax2.plot(time_dist["Time"], time_dist["MinDist"], color="#2ECC71", linewidth=2, label="Min Inter-Drone Dist")

    col_rad = 0.2
    ax2.axhline(y=col_rad, color="red", linestyle="--", linewidth=2, label=f"Collision ({col_rad}m)")
    ax2.fill_between(time_dist["Time"], 0, col_rad, color="red", alpha=0.15)
    ax2.axhline(y=controller.min_dist, color="gray", linestyle="-.", label=f"Baseline ({controller.min_dist}m)")
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlabel("Phase Time (s)")
    ax2.set_ylabel("Distance (m)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(controller.fms_dir, f"{phase_idx:02d}_{phase_name}.png"), dpi=300)
    plt.close()
