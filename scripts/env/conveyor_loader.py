# ============================================================
# 文件: conveyor_loader.py
# ============================================================

import yaml
import numpy as np
from pathlib import Path
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrimView
import time
import os
from pathlib import Path

class ConveyorConfigLoader:
    """传送带配置加载器"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_dir = project_root / "config" / "conveyor_configs"

        self.config_dir = Path(config_dir)
        self.configs = {}
        
        print("=" * 60)
        print("初始化传送带配置加载器")
        print("=" * 60)
        
        self._load_all_configs()
    
    def _load_all_configs(self):
        """加载所有传送带配置文件"""
        
        yaml_files = list(self.config_dir.glob("*.yaml"))
        
        print(f"\n发现 {len(yaml_files)} 个传送带配置文件:")
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
                conveyor_name = config_data['conveyor_config']['conveyor_name']
                self.configs[conveyor_name] = config_data['conveyor_config']
                
                print(f"  ✓ 加载配置: {conveyor_name} ({yaml_file.name})")
        
        print("=" * 60)
    
    def get_config(self, conveyor_name):
        """获取指定传送带的配置"""
        
        config = self.configs.get(conveyor_name)
        
        print(f"\n获取传送带配置: {conveyor_name}")
        print(f"  类型: {config['conveyor_type']}")
        print(f"  尺寸: {config['dimensions']['length']}m x {config['dimensions']['width']}m")
        print(f"  速度: {config['motion']['velocity']} m/s")
        print(f"  最大物体数: {config.get('object_spawning', {}).get('max_objects_on_belt', 'N/A')}")
        
        return config
    
    def list_available_conveyors(self):
        """列出所有可用的传送带"""
        
        print("\n" + "=" * 60)
        print("可用传送带列表:")
        print("=" * 60)
        
        for idx, (name, config) in enumerate(self.configs.items(), 1):
            dims = config['dimensions']
            motion = config['motion']
            print(f"{idx}. {name}")
            print(f"   类型: {config['conveyor_type']}")
            print(f"   长度: {dims['length']:.2f}m")
            print(f"   速度: {motion['velocity']:.2f} m/s (最大: {motion['max_velocity']:.2f} m/s)")
            print(f"   传感器数量: {len(config.get('sensors', []))}")
            print()


class ConveyorFactory:
    """传送带工厂 - 根据配置创建传送带实例"""
    
    def __init__(self, world, assets_root):
        self.world = world
        self.assets_root = assets_root
        self.config_loader = ConveyorConfigLoader()
        self.active_conveyors = {}
        
        print("\n" + "=" * 60)
        print("传送带工厂初始化完成")
        print("=" * 60)
    
    def usd_euler_to_quat(self, euler, degrees=False):
        """欧拉角转四元数"""
        roll, pitch, yaw = [float(e) for e in euler]
        
        if degrees:
            roll = np.radians(roll)
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)
        
        rot_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), roll)
        rot_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch)
        rot_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
        
        total_rot = rot_z * rot_y * rot_x
        quat = total_rot.GetQuat()
        
        return [quat.GetReal(), quat.GetImaginary()[0], 
                quat.GetImaginary()[1], quat.GetImaginary()[2]]
    
    def create_conveyor(self, conveyor_name, instance_name=None, custom_position=None):
        """创建传送带实例"""
        
        print("\n" + "=" * 60)
        print(f"开始创建传送带: {conveyor_name}")
        print("=" * 60)
        
        config = self.config_loader.get_config(conveyor_name)
        
        instance_name = instance_name or f"{conveyor_name}_instance"
        prim_path = f"{config['prim_path']}_{instance_name}"
        
        print(f"\n步骤1: 加载USD资源")
        print(f"  USD路径: {self.assets_root + config['usd_path']}")
        print(f"  Prim路径: {prim_path}")
        
        add_reference_to_stage(
            usd_path=self.assets_root + config['usd_path'],
            prim_path=prim_path
        )
        
        print(f"\n步骤2: 配置物理属性")
        self._setup_conveyor_physics(prim_path, config)
        
        print(f"\n步骤3: 设置位姿")
        position = custom_position if custom_position else config['root_position']
        print(f"  位置: {position}")
        print(f"  姿态(欧拉角): {config['root_orientation']}°")
        
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        
        for op in ops:
            if op.GetOpName() == "xformOp:translate":
                op.Set(Gf.Vec3d(position))
            elif op.GetOpName() == "xformOp:orient":
                quat_list = self.usd_euler_to_quat(config['root_orientation'], degrees=True)
                quat = Gf.Quatd(quat_list[0], quat_list[1], quat_list[2], quat_list[3])
                op.Set(quat)
        
        print(f"\n步骤4: 配置运动参数")
        self._setup_conveyor_motion(prim_path, config)
        
        print(f"\n步骤5: 设置区域标记")
        self._setup_conveyor_zones(config, position)
        
        print(f"\n步骤6: 配置传感器")
        sensor_prims = self._setup_conveyor_sensors(config, position)
        
        conveyor_instance = {
            'prim_path': prim_path,
            'config': config,
            'position': position,
            'sensor_prims': sensor_prims,
            'is_running': False,
            'current_velocity': 0.0,
            'objects_on_belt': [],
            'spawn_timer': 0.0
        }
        
        self.active_conveyors[instance_name] = conveyor_instance
        
        print("\n" + "=" * 60)
        print(f"✓ 传送带 {conveyor_name} 创建完成!")
        print("=" * 60)
        
        return conveyor_instance
    
    def _setup_conveyor_physics(self, prim_path, config):
        """设置传送带物理属性"""
        
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        
        physics_config = config['physics']
        print(f"  表面材料: {physics_config['surface_material']}")
        print(f"  静摩擦系数: {physics_config['belt_friction']['static_friction']}")
        print(f"  动摩擦系数: {physics_config['belt_friction']['dynamic_friction']}")
        
        UsdPhysics.CollisionAPI.Apply(prim)
        
        material_api = UsdPhysics.MaterialAPI.Apply(prim)
        material_api.CreateStaticFrictionAttr().Set(physics_config['belt_friction']['static_friction'])
        material_api.CreateDynamicFrictionAttr().Set(physics_config['belt_friction']['dynamic_friction'])
        material_api.CreateRestitutionAttr().Set(physics_config['belt_friction']['restitution'])
    
    def _setup_conveyor_motion(self, prim_path, config):
        """设置传送带运动"""
        
        motion_config = config['motion']
        print(f"  初始速度: {motion_config['velocity']} m/s")
        print(f"  运动方向: {motion_config['direction']}")
        print(f"  加速度: {motion_config['acceleration']} m/s²")
        print(f"  可反向: {motion_config['can_reverse']}")
    
    def _setup_conveyor_zones(self, config, base_position):
        """设置传送带区域"""
        
        zones = config.get('zones', {})
        print(f"  定义区域数: {len(zones)}")
        
        for zone_name, zone_data in zones.items():
            print(f"    - {zone_name}: {zone_data.get('start', 0):.2f}m - {zone_data.get('end', 0):.2f}m")
    
    def _setup_conveyor_sensors(self, config, base_position):
        """设置传送带传感器"""
        
        sensors = config.get('sensors', [])
        print(f"  传感器数量: {len(sensors)}")
        
        sensor_prims = []
        for sensor in sensors:
            print(f"    - {sensor['name']}: {sensor['sensor_type']} @ {sensor['position']}")
            sensor_prims.append({
                'name': sensor['name'],
                'type': sensor['sensor_type'],
                'position': sensor['position'],
                'range': sensor.get('detection_range', 0.3)
            })
        
        return sensor_prims
    
    def start_conveyor(self, instance_name):
        """启动传送带"""
        
        conveyor = self.active_conveyors.get(instance_name)
        
        print(f"\n===== 启动传送带: {instance_name}")
        conveyor['is_running'] = True
        conveyor['current_velocity'] = conveyor['config']['motion']['velocity']
        print(f"✓ 传送带运行中，速度: {conveyor['current_velocity']} m/s")
    
    def stop_conveyor(self, instance_name):
        """停止传送带"""
        
        conveyor = self.active_conveyors.get(instance_name)
        
        print(f"\n===== 停止传送带: {instance_name}")
        conveyor['is_running'] = False
        conveyor['current_velocity'] = 0.0
        print(f"✓ 传送带已停止")
    
    def set_conveyor_speed(self, instance_name, speed):
        """设置传送带速度"""
        
        conveyor = self.active_conveyors.get(instance_name)
        
        max_speed = conveyor['config']['motion']['max_velocity']
        speed = np.clip(speed, 0.0, max_speed)
        
        print(f"\n===== 设置传送带速度: {instance_name}")
        print(f"目标速度: {speed:.2f} m/s (限制: 0 - {max_speed:.2f})")
        
        conveyor['current_velocity'] = speed
        conveyor['config']['motion']['velocity'] = speed
        
        print(f"✓ 速度已更新")
    
    def update_objects_on_belt(self, instance_name, dt):
        """更新传送带上物体的运动"""
        
        conveyor = self.active_conveyors.get(instance_name)
        
        if not conveyor['is_running']:
            return
        
        velocity = conveyor['current_velocity']
        direction = np.array(conveyor['config']['motion']['direction'])
        
        displacement = direction * velocity * dt
        
        for obj in conveyor['objects_on_belt']:
            pass


def demo_conveyor_usage():
    """演示如何使用传送带工厂"""
    
    from omni.isaac.core import World
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    
    world = World(stage_units_in_meters=1.0)
    assets_root = get_assets_root_path()
    
    print("\n" + "=" * 60)
    print("码垛系统传送带管理演示")
    print("=" * 60)
    
    factory = ConveyorFactory(world, assets_root)
    
    factory.config_loader.list_available_conveyors()
    
    print("\n" + "=" * 60)
    print("场景1: 创建标准传送带")
    print("=" * 60)
    
    conveyor1 = factory.create_conveyor("standard_belt_small", "main_conveyor")
    
    print("\n" + "=" * 60)
    print("场景2: 创建模块化传送带系统")
    print("=" * 60)
    
    conveyor2 = factory.create_conveyor("modular_belt_system", "assembly_line")
    
    print("\n" + "=" * 60)
    print("场景3: 控制传送带")
    print("=" * 60)
    
    factory.start_conveyor("main_conveyor")
    time.sleep(0.5)
    factory.set_conveyor_speed("main_conveyor", 0.3)
    time.sleep(0.5)
    factory.stop_conveyor("main_conveyor")
    
    print("\n" + "=" * 60)
    print("场景4: 访问传送带信息")
    print("=" * 60)
    
    print(f"\n主传送带传感器:")
    for sensor in conveyor1['sensor_prims']:
        print(f"  {sensor['name']}: {sensor['type']} (range: {sensor['range']}m)")
    
    print(f"\n装配线区域:")
    for zone_name in conveyor2['config']['zones'].keys():
        print(f"  - {zone_name}")
    
    print("\n" + "=" * 60)
    print("传送带系统就绪，可以开始码垛!")
    print("=" * 60)


if __name__ == "__main__":
    demo_conveyor_usage()