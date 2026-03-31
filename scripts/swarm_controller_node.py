#!/usr/bin/env python3

from flock_gpt.msg import Vector3StampedArray
from geometry_msgs.msg import Vector3
import numpy as np
import rospy
from sdf import box, sphere, write_binary_stl, rounded_box, capsule
from sdf import generate
from apf_controller import APFSwarmController
from gpt_sdf import SDFDialog, SDFModel, CURRENT_MODEL
import threading
import os
import time
import sys  # 🌟 新增：用于刷新缓冲区和检测终端状态
from point_distributor import PointDistributer

class SwarmControllerNode():
    def get_input(self, prompt, default_val):
        """🛡️ 工业级输入捕获：防止 roslaunch 后台模式导致的 EOFError 崩溃"""
        sys.stdout.flush()  # 强制刷新缓冲区，确保打印能立刻显示
        try:
            # 检测是否处于无交互终端环境 (例如标准的 roslaunch)
            if not sys.stdin.isatty():
                print(f"{prompt}\n[!] ⚠️ 非交互模式 (roslaunch) 运行中，自动套用默认值: {default_val}")
                sys.stdout.flush()
                return str(default_val)
            
            val = input(prompt)
            return val if val.strip() != "" else str(default_val)
        except (EOFError, KeyboardInterrupt):
            print(f"\n[!] ⚠️ 检测到输入流异常 (可能是 roslaunch 导致)，自动套用默认值: {default_val}")
            sys.stdout.flush()
            return str(default_val)

    def should_generate_fms_plot(self, phase_desc):
        """会话级总开关：启动时决定是否全程生成 FLSM 图。"""
        return self.enable_fms_srm_plots

    def __init__(self, goals=[]) -> None:
        # ⚛️ 系统启动界面（FlockGPT-AFS）
        print("\n" + "═"*65)
        print("   ⚛️  FlockGPT-AFS Framework Initializing...")
        print("   (Autonomous Resilient & Elastic Swarm)")
        print("   -------------------------------------------------")
        # 👇 新增下面这一行，动态显示大模型名称
        print(f"   [Loaded] LLM : {CURRENT_MODEL} (Shape Generation Engine)") 
        print("   [Loaded] FLSM: Fleet Lifecycle Safety Module")
        print("   [Standby] DCA: Dynamic Configuration Assignment")
        print("═"*65)
        sys.stdout.flush()
        
        # 🌟 核心理念：设定最大机队容量（无人机池）
        val = self.get_input(">>> [FLSM] Enter Maximum Fleet Capacity (e.g. 200, must be >= max shape size): ", "200")
        try:
            self.fleet_capacity = int(val)
        except:
            self.fleet_capacity = 200
            
        # 设置模拟器一次性生成，后续绝不修改此参数，彻底杜绝重启
        rospy.set_param('/swarm_num_drones', self.fleet_capacity)
        
        self.shape_drones = 0       # 当前形状所需无人机数
        self.prev_active_drones = 0 # 记录上一轮飞在空中的无人机数

        print("\n" + "━"*60)
        print("--- ⚙️ Algorithm Module Configuration ---")
        module_input = self.get_input(">>> [DCA] Enable Dynamic Configuration Assignment? [y/N]: ", "n")
        self.enable_dca = True if module_input.lower() == 'y' else False
        fms_plot_input = self.get_input(">>> [FLSM] Enable plot generation for the whole run? [y/N]: ", "n")
        self.enable_fms_srm_plots = (fms_plot_input.strip().lower() == 'y')
        
        # 🎯 动态安全基线设定 (升级为 0.35m 默认，0.3-1.0m 范围)
        self.safety_baseline = 0.35
        if self.enable_dca:
            sb_input = self.get_input(">>> [DCA] Set Safety Baseline Distance (0.3 - 1.0m, Default 0.35m): ", "0.35")
            try:
                val = float(sb_input)
                if 0.3 <= val <= 1.0:
                    self.safety_baseline = val
                else:
                    print(f"[!] [DCA] Out of bounds ({val}). Defaulting to 0.35m.")
            except ValueError:
                print("[!] [DCA] Invalid input. Defaulting to 0.35m.")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 🌟 分离 DCA 图表和 FLSM 轨迹图的文件夹
        self.save_dir = os.path.join(base_dir, 'DCA-result', timestamp)
        self.fms_dir = os.path.join(base_dir, 'FLSM-result', timestamp) if self.enable_fms_srm_plots else ""
        os.makedirs(self.save_dir, exist_ok=True)
        if self.enable_fms_srm_plots:
            os.makedirs(self.fms_dir, exist_ok=True)
        
        self.phase_idx = 1 # 🌟 新增：多轮次变换的阶段计数器
        
        print("\n" + "="*60)
        print(f"[*] [FlockGPT-AFS] Mode: DCA Enabled = {self.enable_dca}")
        print(f"[*] [FlockGPT-AFS] Safety Baseline : {self.safety_baseline}m")
        print(f"[*] [FLSM] Fleet Capacity: {self.fleet_capacity} drones (Simulating...)")
        print("="*60 + "\n")
        sys.stdout.flush()

        self.is_running = False 
        self.goals = []
        self.start_poses = None
        self.home_poses = None  
        self.trigger_return = False 
        
        self.controller = APFSwarmController(max_vel=0.9, min_dist=self.safety_baseline)
        self.controller.log_dir = self.save_dir
        self.controller.fms_dir = self.fms_dir  # 🌟 挂载 FLSM 专属输出目录
        self.controller.enable_dca = self.enable_dca
        
        self.model = SDFModel()
        self.dialog = SDFDialog()

        rospy.Subscriber("/swarm/poses", Vector3StampedArray, self.callback_state, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher('/swarm/cmd_vel', Vector3StampedArray, queue_size=1)
        
        threading.Thread(target=self.continuous_input_prompt, daemon=True).start()
        # 🌟 注册关机清屏钩子
        rospy.on_shutdown(self.cleanup_environment)

    def cleanup_environment(self):
        """🚀 当节点被正常结束 或 Ctrl+C 强制退出时，自动触发"""
        print("\n[*] 🧹 [FLSM] Cleaning up environment... Clearing Unity screen.")
        rospy.set_param('/swarm_num_drones', 0) # 告诉 test.py 清屏
        sys.stdout.flush()
        
    def callback_state(self, msg:Vector3StampedArray):
        poses = np.array([[p.x, p.y, p.z] for p in msg.vector])
        
        # 🌟 获取模拟器中全部的机队网格坐标，作为永久的大本营
        if self.home_poses is None and len(poses) >= self.fleet_capacity:
            self.home_poses = poses[:self.fleet_capacity].copy()
            self.controller.global_home_poses = self.home_poses.copy() # 🌟 新增：把全局基地给控制器备份一份用来画图
            print(f"[*] [FLSM] Captured home grid for {self.fleet_capacity} drones. Fleet ready for dispatch.")
            # 🌟 新增：触发绘制 "第 0 张：待命网格灰点图"
            if self.should_generate_fms_plot("FLSM Standby (Phase 0)"):
                self.controller.generate_idle_report(self.home_poses)
            sys.stdout.flush()
            
        if getattr(self, 'trigger_return', False):
            self.controller.initiate_safe_return(poses, self.home_poses)
            self.trigger_return = False
            self.start_poses = poses 
        
        if not self.is_running or self.goals is None or np.array(self.goals).size == 0:
            return
            
        if self.start_poses is None:
            if self.home_poses is None: return # 等待环境加载

            # 顺序增减编：仅覆盖“上一轮完成数量”和“本轮目标数量”的并集
            runtime_active = min(max(self.prev_active_drones, self.shape_drones), self.fleet_capacity)
            
            self.controller.distribute_goals(
                poses, 
                self.goals, 
                shape_num=self.shape_drones, 
                active_num=runtime_active
            ) 
            self.start_poses = poses
            

        vels = self.controller.get_control(poses)
        cmd_vel = Vector3StampedArray()
        
        for vel in vels:
            vect = Vector3()
            vect.x, vect.y, vect.z = vel[0], vel[1], vel[2]
            cmd_vel.vector.append(vect)
        
        if not rospy.is_shutdown():
            self.cmd_vel_publisher.publish(cmd_vel)

    def process_user_input(self, user_input):
        sdf_code = self.dialog.get_next_sdf_code(user_input)
        if sdf_code:
            local_vars = {"f": None}
            try: 
                f=10
                exec(sdf_code, globals(), local_vars)  
                f = local_vars.get("f")
                if f is not None:
                    pd_dist = PointDistributer(f)
                    points = pd_dist.generate_points(self.shape_drones)

                    # 🚀 ==========================================
                    # 补回丢失的升空逻辑：让表演机队对齐中心并飞向 8 米高空！
                    # ==========================================
                    if self.home_poses is not None and len(self.home_poses) > 0:
                        home_center = np.mean(self.home_poses, axis=0)
                        points[:, 0] += home_center[0] # X轴对齐停机坪中心
                        points[:, 1] += home_center[1] # Y轴对齐停机坪中心
                        points[:, 2] += 8.0            # Z轴(高度)强行拔高 8 米！
                    # ==========================================
                    
                    # 🌟 池化路由分配：构造全局目标阵列
                    full_goals = np.zeros((self.fleet_capacity, 3))
                    
                    # 1. 构图部队
                    if self.shape_drones > 0:
                        full_goals[:self.shape_drones] = points[:self.shape_drones]
                        
                    # 2. 待命/返航部队
                    if self.fleet_capacity > self.shape_drones:
                        full_goals[self.shape_drones:] = self.home_poses[self.shape_drones:]
                        
                    self.goals = full_goals
            except Exception as e:
                print("Error: ", e)

    def execute_return_sequence(self):
        if self.home_poses is None:
            print("🛑 [FlockGPT-AFS] Shutting down system...")
            sys.stdout.flush()
            rospy.signal_shutdown("User exit")
            return
            
        print("\n" + "="*60)
        print("[*] 🛬 [FLSM] Activating global return...")
        print("[*] [FLSM] Executing global return for all airborne drones.")
        print("="*60)
        sys.stdout.flush()
        
        self.trigger_return = True
        self.controller.trajectory_log.clear()       # 🌟 新增：清空准备记录纯返航轨迹
        self.controller.phase_start_time = time.time() # 🌟 新增
        self.controller.current_log_name = "" 
        prev_active = int(self.prev_active_drones)
        self.controller.phase_prev_active_num = prev_active
        self.controller.phase_shape_num = 0
        self.controller.phase_active_num = prev_active
        self.controller.phase_new_launch_ids = np.array([], dtype=int)
        self.controller.phase_return_ids = np.arange(prev_active, dtype=int) if prev_active > 0 else np.array([], dtype=int)
        self.is_running = True
        
        self.get_input("\n>>> [FLSM] Press 'Enter' when all drones have landed safely to power off...", "")
        self.is_running = False
        # 🌟 新增：生成全员落地灰点图
        if self.should_generate_fms_plot("FLSM Global Return"):
            self.controller.generate_fms_srm_report("Global_Return", self.phase_idx)
        print("🛑 [FlockGPT-AFS] System powered off successfully.")
        sys.stdout.flush()
        rospy.signal_shutdown("Experiment finished")

    def continuous_input_prompt(self):
        rospy.sleep(1.0)
        
        while not rospy.is_shutdown():
            # 动态请求无人机数量
            num_input = self.get_input(f"\n>>> [FLSM] Enter target shape size (Max {self.fleet_capacity}): ", "10")
            try:
                self.shape_drones = min(int(num_input), self.fleet_capacity)
            except:
                print("[!] [FLSM] Invalid number. Setting to 10.")
                self.shape_drones = 10
                
            user_input = self.get_input(f">>> [FlockGPT-AFS] What to build with {self.shape_drones} drones? (e.g., sphere) [type 'exit' to quit]: \n", "exit")
            if user_input.lower() in ['exit', 'quit']:
                self.execute_return_sequence() 
                break
            
            shape_name = user_input.replace(" ", "_")
            mode_str = "DCA" if self.enable_dca else "Base"
            
            # 💡 将安全基线嵌入文件名，供比对脚本读取
            sd_str = f"_SD{self.controller.min_dist:.2f}"
            self.controller.current_log_name = f"data_{shape_name}_{self.shape_drones}drones_{mode_str}{sd_str}"
            
            self.controller.is_returning = False
            self.process_user_input(user_input)
            self.controller.trajectory_log.clear()
            self.controller.phase_start_time = time.time()
            prev_active = int(self.prev_active_drones)
            curr_shape = int(self.shape_drones)
            runtime_active = int(min(max(prev_active, curr_shape), self.fleet_capacity))
            new_launch_ids = np.arange(prev_active, runtime_active, dtype=int) if runtime_active > prev_active else np.array([], dtype=int)
            return_ids = np.arange(curr_shape, prev_active, dtype=int) if prev_active > curr_shape else np.array([], dtype=int)
            self.controller.phase_prev_active_num = prev_active
            self.controller.phase_shape_num = curr_shape
            self.controller.phase_active_num = runtime_active
            self.controller.phase_new_launch_ids = new_launch_ids
            self.controller.phase_return_ids = return_ids
            self.start_poses = None
            self.start_poses = None
            self.is_running = True 
            
            # 极其优雅的调度提示
            if self.shape_drones > self.prev_active_drones:
                print(f"[*] [FLSM] Scaling UP: {self.prev_active_drones} airborne + {self.shape_drones - self.prev_active_drones} launching from ground.")
            elif self.shape_drones < self.prev_active_drones:
                print(f"[*] [FLSM] Scaling DOWN: {self.shape_drones} morphing shape, {self.prev_active_drones - self.shape_drones} automatically returning to base.")
            else:
                print(f"[*] [FLSM] Seamless Morphing: All {self.shape_drones} airborne drones transitioning to new shape.")
                
            print(f"[*] [FlockGPT-AFS] Deploying '{shape_name}' (Log: {self.controller.current_log_name}.csv)...")
            sys.stdout.flush()

            self.get_input("\n>>> [FlockGPT-AFS] Press 'Enter' when formation is complete to generate individual plots...", "")
            self.is_running = False 
            # 用户确认当前阶段完成后，更新上一轮完成数量
            self.prev_active_drones = self.shape_drones
            
            self.controller.generate_plots()
            # 🌟 新增：生成 FLSM 轨迹图
            if self.should_generate_fms_plot(f"Phase {self.phase_idx}: {shape_name}"):
                self.controller.generate_fms_srm_report(shape_name, self.phase_idx)
            self.phase_idx += 1
            
            cont = self.get_input("\n>>> [FlockGPT-AFS] Do you want to try another shape? (y/n): ", "n")
            if cont.lower() != 'y':
                self.execute_return_sequence() 
                break

            print("\n" + "━"*60)
            print("--- ⚙️ Algorithm Module Configuration (New Round) ---")
            module_input = self.get_input(f">>> [DCA] Enable Dynamic Configuration Assignment? (Current: {self.enable_dca}) [y/N]: ", "n")
            self.enable_dca = True if module_input.lower() == 'y' else False
            self.controller.enable_dca = self.enable_dca

            # 🎯 循环中的动态安全基线更新 (升级为 0.35m 默认，0.3-1.0m 范围)
            if self.enable_dca:
                sb_input = self.get_input(f">>> [DCA] Set Safety Baseline Distance (0.3 - 1.0m, Current: {self.controller.min_dist}m): ", str(self.controller.min_dist))
                try:
                    val = float(sb_input)
                    if 0.3 <= val <= 1.0:
                        self.controller.min_dist = val
                    else:
                        print(f"[!] [DCA] Out of bounds. Keeping {self.controller.min_dist}m.")
                except ValueError:
                    print(f"[!] [DCA] Invalid input. Keeping {self.controller.min_dist}m.")
            else:
                self.controller.min_dist = 0.35 # Baseline 强制锁死 0.35

            print(f"[*] [FlockGPT-AFS] Mode updated: DCA Enabled={self.enable_dca}, Safety Baseline={self.controller.min_dist}m")
            sys.stdout.flush()

            self.goals = []          
            self.start_poses = None  
            rospy.sleep(0.5) 

if __name__ == "__main__": 
    try:
        rospy.init_node('swarm_controller_node', anonymous=True)
        controller = SwarmControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass