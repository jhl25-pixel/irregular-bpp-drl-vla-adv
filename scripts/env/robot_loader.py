# ============================================================
# 文件: robot_loader.py
# ============================================================

import yaml
import numpy as np
from pathlib import Path
from pxr import Gf, UsdGeom
import omni.usd
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
import os
from pathlib import Path

class RobotConfigLoader:
    """机器人配置加载器"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            # 自动找到项目根目录的config文件夹
            current_file = Path(__file__).resolve()  # 当前文件路径
            project_root = current_file.parent.parent.parent  # 上两级到项目根目录
            config_dir = project_root / "config" / "robot_configs"

        self.config_dir = Path(config_dir)
        self.configs = {}
        
        print("=" * 60)
        print("初始化机器人配置加载器")
        print("=" * 60)
        
        self._load_all_configs()
    
    def _load_all_configs(self):
        """加载所有机器人配置文件"""
        
        yaml_files = list(self.config_dir.glob("*.yaml"))
        
        print(f"\n发现 {len(yaml_files)} 个机器人配置文件:")
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
                robot_name = config_data['robot_config']['robot_name']
                self.configs[robot_name] = config_data['robot_config']
                
                print(f"  ✓ 加载配置: {robot_name} ({yaml_file.name})")
        
        print("=" * 60)
    
    def get_config(self, robot_name):
        """获取指定机器人的配置"""
        
        config = self.configs.get(robot_name)
        
        print(f"\n获取机器人配置: {robot_name}")
        print(f"  制造商: {config['manufacturer']}")
        print(f"  自由度: {config['joints']['dof']}")
        print(f"  末端执行器: {config['end_effector']['type']}")
        
        return config
    
    def list_available_robots(self):
        """列出所有可用的机器人"""
        
        print("\n" + "=" * 60)
        print("可用机器人列表:")
        print("=" * 60)
        
        for idx, (name, config) in enumerate(self.configs.items(), 1):
            print(f"{idx}. {name}")
            print(f"   制造商: {config['manufacturer']}")
            print(f"   类型: {config['robot_type']}")
            print(f"   自由度: {config['joints']['dof']}")
            print(f"   负载: {config['physics']['payload_capacity']} kg")
            print()


class RobotFactory:
    """机器人工厂"""
    
    def __init__(self, world, assets_root):
        self.world = world
        self.assets_root = assets_root
        self.config_loader = RobotConfigLoader()
        self.robots = {}  # 存储已创建的机器人
        
        print("\n" + "=" * 60)
        print("机器人工厂初始化完成")
        print("=" * 60)
    
    def create_robot(self, robot_name, instance_name, position_override=None, orientation_override=None):
        """创建机器人
        
        Args:
            robot_name: 机器人配置名称
            instance_name: 实例名称
            position_override: 覆盖配置中的位置 (可选)
            orientation_override: 覆盖配置中的姿态 (可选)
        """
        print("\n" + "=" * 60)
        print(f"开始创建机器人: {robot_name}")
        print("=" * 60)
        
        # 获取配置
        config = self.config_loader.get_config(robot_name)
        if not config:
            print(f"✗ 未找到机器人配置: {robot_name}")
            return None, None
        
        # 步骤1: 加载USD
        print("\n步骤1: 加载USD资源")
        usd_path = config['usd_path']
        full_path = self.assets_root + usd_path
        prim_path = f"/World/{config['prim_path'].split('/')[-1]}_{instance_name}"
        
        print(f"  USD路径: {full_path}")
        print(f"  Prim路径: {prim_path}")
        
        add_reference_to_stage(usd_path=full_path, prim_path=prim_path)
        
        # 步骤2: 配置末端执行器
        print("\n步骤2: 配置末端执行器")
        ee_config = config['end_effector']
        print(f"  类型: {ee_config['type']}")
        print(f"  关节: {ee_config['joint_names']}")
        
        gripper = self._create_gripper(prim_path, ee_config)
        
        # 步骤3: 创建机械臂
        print("\n步骤3: 创建机械臂对象")
        robot = self._create_manipulator(prim_path, instance_name, gripper, config)
        
        # 步骤4: 设置位姿
        print("\n步骤4: 设置基座位姿")
        final_position = position_override if position_override is not None else config['root_position']
        final_orientation = orientation_override if orientation_override is not None else config['root_orientation']
        
        print(f"  位置: {final_position}")
        print(f"  姿态(欧拉角): {final_orientation}°")
        
        self._set_robot_transform(prim_path, final_position, final_orientation)
        
        print("\n" + "=" * 60)
        print(f"✓ 机器人 {robot_name} 创建完成!")
        print("=" * 60)
        
        # 保存机器人实例
        self.robots[instance_name] = {
            'robot': robot,
            'config': config,
            'prim_path': prim_path
        }
        
        return robot, config
    
    def _create_gripper(self, robot_prim_path, ee_config):
        """创建夹爪"""
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        
        ee_prim_path = f"{robot_prim_path}/{ee_config['prim_name']}"
        
        gripper = ParallelGripper(
            end_effector_prim_path=ee_prim_path,
            joint_prim_names=ee_config['joint_names'],
            joint_opened_positions=np.array(ee_config['open_positions']),
            joint_closed_positions=np.array(ee_config['close_positions']),
            action_deltas=np.array(ee_config.get('action_deltas', [0.01, 0.01])),
        )
        
        return gripper
    
    def _create_manipulator(self, prim_path, name, gripper, config):
        """创建机械臂"""
        from isaacsim.robot.manipulators import SingleManipulator
        
        robot = self.world.scene.add(
            SingleManipulator(
                prim_path=prim_path,
                name=name,
                end_effector_prim_name=config['end_effector']['prim_name'].split('/')[-1],
                gripper=gripper,
            )
        )
        
        return robot
    
    def _set_robot_transform(self, prim_path, position, orientation_euler):
        """设置机器人的位姿
        
        Args:
            prim_path: 机器人的Prim路径
            position: 位置 [x, y, z]
            orientation_euler: 欧拉角姿态 [roll, pitch, yaw] (度)
        """
        import omni.usd
        from pxr import UsdGeom, Gf
        
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        
        if not prim:
            print(f"  ⚠️  警告: 未找到Prim {prim_path}")
            return
        
        xformable = UsdGeom.Xformable(prim)
        
        # 获取或创建变换操作
        ops = xformable.GetOrderedXformOps()
        
        # 如果没有变换操作，创建它们
        if not ops:
            xformable.AddTranslateOp()
            xformable.AddOrientOp()
            xformable.AddScaleOp()
            ops = xformable.GetOrderedXformOps()
        
        # 设置位置和旋转
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(position))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                # 将欧拉角(度)转换为四元数
                quat = self._euler_to_quaternion(orientation_euler)
                op.Set(Gf.Quatd(quat[0], quat[1], quat[2], quat[3]))
        
        print(f"  ✓ 变换设置完成")
    
    def _euler_to_quaternion(self, euler_degrees):
        """欧拉角(度)转四元数 [w, x, y, z]
        
        Args:
            euler_degrees: [roll, pitch, yaw] in degrees
            
        Returns:
            [w, x, y, z] quaternion
        """
        from pxr import Gf
        
        # 转换为弧度
        roll = np.radians(euler_degrees[0])
        pitch = np.radians(euler_degrees[1])
        yaw = np.radians(euler_degrees[2])
        
        # 使用USD的Rotation类
        rot_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), roll)
        rot_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch)
        rot_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
        
        # 组合旋转
        total_rot = rot_z * rot_y * rot_x
        quat = total_rot.GetQuat()
        
        # 返回 [w, x, y, z]
        return [
            quat.GetReal(),
            quat.GetImaginary()[0],
            quat.GetImaginary()[1],
            quat.GetImaginary()[2]
        ]
    
    def get_robot(self, instance_name):
        """获取已创建的机器人实例"""
        return self.robots.get(instance_name)
    
    def list_created_robots(self):
        """列出已创建的机器人"""
        print("\n" + "=" * 60)
        print("已创建的机器人:")
        print("=" * 60)
        
        if not self.robots:
            print("  (无)")
            return
        
        for name, info in self.robots.items():
            print(f"\n  • {name}")
            print(f"    类型: {info['config']['robot_name']}")
            print(f"    路径: {info['prim_path']}")


# ============================================================
# 使用示例
# ============================================================

def demo_usage():
    """演示如何使用机器人工厂"""
    
    from omni.isaac.core import World
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    
    world = World(stage_units_in_meters=1.0)
    assets_root = get_assets_root_path()
    
    print("\n" + "=" * 60)
    print("码垛系统机器人管理演示")
    print("=" * 60)
    
    factory = RobotFactory(world, assets_root)
    
    factory.config_loader.list_available_robots()
    
    print("\n" + "=" * 60)
    print("场景1: 创建单个Franka Panda机器人")
    print("=" * 60)
    
    panda_arm, panda_config = factory.create_robot("franka_panda", "robot_A")
    
    print("\n" + "=" * 60)
    print("场景2: 创建多个不同型号的机器人")
    print("=" * 60)
    
    fr3_arm, fr3_config = factory.create_robot("franka_fr3", "robot_B")
    ur5e_arm, ur5e_config = factory.create_robot("ur5e", "robot_C")
    
    print("\n" + "=" * 60)
    print("场景3: 访问机器人配置信息")
    print("=" * 60)
    
    print(f"\nPanda工作空间: {panda_config['workspace']}")
    print(f"UR5e关节限制: {ur5e_config['joints']['velocity_limits']}")
    print(f"FR3负载能力: {fr3_config['physics']['payload_capacity']} kg")
    
    print("\n" + "=" * 60)
    print("所有机器人已就绪，可以开始码垛任务!")
    print("=" * 60)


if __name__ == "__main__":
    demo_usage()