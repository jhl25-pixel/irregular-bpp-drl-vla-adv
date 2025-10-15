# ============================================================
# 文件: box_loader.py
# ============================================================

import yaml
import numpy as np
from pathlib import Path
from pxr import Gf, UsdGeom, UsdPhysics
import omni.usd
from omni.isaac.core.utils.stage import add_reference_to_stage
import os
from pathlib import Path

class BoxConfigLoader:
    """箱子配置加载器"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_dir = project_root / "config" / "object_configs"

        self.config_dir = Path(config_dir)
        self.configs = {}
        
        print("=" * 60)
        print("初始化箱子配置加载器")
        print("=" * 60)
        
        self._load_all_configs()
    
    def _load_all_configs(self):
        """加载所有箱子配置文件"""
        
        yaml_files = list(self.config_dir.glob("*.yaml"))
        
        print(f"\n发现 {len(yaml_files)} 个箱子配置文件:")
        
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
                box_name = config_data['box_config']['box_name']
                self.configs[box_name] = config_data['box_config']
                
                print(f"  ✓ 加载配置: {box_name} ({yaml_file.name})")
        
        print("=" * 60)
    
    def get_config(self, box_name):
        """获取指定箱子的配置"""
        
        config = self.configs.get(box_name)
        
        print(f"\n获取箱子配置: {box_name}")
        print(f"  类型: {config['box_type']}")
        print(f"  尺寸: {config['dimensions']['length']}m x {config['dimensions']['width']}m x {config['dimensions']['height']}m")
        print(f"  最大层数: {config['stacking']['max_layers']}")
        
        return config
    
    def list_available_boxes(self):
        """列出所有可用的箱子"""
        
        print("\n" + "=" * 60)
        print("可用箱子列表:")
        print("=" * 60)
        
        for idx, (name, config) in enumerate(self.configs.items(), 1):
            dims = config['dimensions']
            stack = config['stacking']
            print(f"{idx}. {name}")
            print(f"   类型: {config['box_type']}")
            print(f"   尺寸: {dims['length']:.2f}m x {dims['width']:.2f}m x {dims['height']:.2f}m")
            print(f"   质量: {config['physics']['mass']} kg")
            print(f"   码垛网格: {stack['grid_size'][0]}x{stack['grid_size'][1]} ({stack['max_layers']}层)")
            print()


class BoxFactory:
    """箱子工厂 - 根据配置创建箱子实例"""
    
    def __init__(self, world, assets_root):
        self.world = world
        self.assets_root = assets_root
        self.config_loader = BoxConfigLoader()
        
        print("\n" + "=" * 60)
        print("箱子工厂初始化完成")
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
    
    def create_box(self, box_name, instance_name=None, custom_position=None):
        """创建箱子实例"""
        
        print("\n" + "=" * 60)
        print(f"开始创建箱子: {box_name}")
        print("=" * 60)
        
        config = self.config_loader.get_config(box_name)
        
        instance_name = instance_name or f"{box_name}_instance"
        prim_path = f"{config['prim_path']}_{instance_name}"
        
        print(f"\n步骤1: 加载USD资源")
        print(f"  USD路径: {self.assets_root + config['usd_path']}")
        print(f"  Prim路径: {prim_path}")
        
        add_reference_to_stage(
            usd_path=self.assets_root + config['usd_path'],
            prim_path=prim_path
        )
        
        print(f"\n步骤2: 配置物理属性")
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        
        physics_config = config['physics']
        print(f"  质量: {physics_config['mass']} kg")
        print(f"  静态物体: {physics_config['is_static']}")
        print(f"  静摩擦系数: {physics_config['friction']['static_friction']}")
        
        if physics_config['is_static']:
            pass
        else:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            rigid_api.CreateRigidBodyEnabledAttr(True)
        
        UsdPhysics.CollisionAPI.Apply(prim)
        collision_api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        collision_api.CreateCollisionEnabledAttr(physics_config['enable_collision'])
        
        material_api = UsdPhysics.MaterialAPI.Apply(prim)
        material_api.CreateStaticFrictionAttr().Set(physics_config['friction']['static_friction'])
        material_api.CreateDynamicFrictionAttr().Set(physics_config['friction']['dynamic_friction'])
        material_api.CreateRestitutionAttr().Set(physics_config['friction']['restitution'])
        
        print(f"\n步骤3: 设置位姿")
        position = custom_position if custom_position else config['root_position']
        print(f"  位置: {position}")
        print(f"  姿态(欧拉角): {config['root_orientation']}°")
        
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        
        for op in ops:
            if op.GetOpName() == "xformOp:translate":
                op.Set(Gf.Vec3d(position))
            elif op.GetOpName() == "xformOp:orient":
                quat_list = self.usd_euler_to_quat(config['root_orientation'], degrees=True)
                quat = Gf.Quatd(quat_list[0], quat_list[1], quat_list[2], quat_list[3])
                op.Set(quat)
        
        print(f"\n步骤4: 计算码垛位置")
        self._calculate_stacking_positions(config, position)
        
        print("\n" + "=" * 60)
        print(f"✓ 箱子 {box_name} 创建完成!")
        print("=" * 60)
        
        return prim_path, config
    
    def _calculate_stacking_positions(self, config, box_position):
        """计算箱子内的码垛位置"""
        
        dims = config['internal_dimensions']
        stack = config['stacking']
        
        grid_x, grid_y = stack['grid_size']
        spacing = stack['object_spacing']
        layer_height = stack['layer_height']
        max_layers = stack['max_layers']
        
        cell_size_x = (dims['length'] - spacing * (grid_x + 1)) / grid_x
        cell_size_y = (dims['width'] - spacing * (grid_y + 1)) / grid_y
        
        print(f"  码垛网格: {grid_x} x {grid_y}")
        print(f"  每层可放置: {grid_x * grid_y} 个物体")
        print(f"  总容量: {grid_x * grid_y * max_layers} 个物体")
        print(f"  单元格尺寸: {cell_size_x:.3f}m x {cell_size_y:.3f}m")
        
        stacking_positions = []
        
        box_x, box_y, box_z = box_position
        
        for layer in range(max_layers):
            z = box_z + dims['wall_thickness'] + layer * layer_height + layer_height / 2
            
            for i in range(grid_x):
                for j in range(grid_y):
                    x = box_x - dims['length']/2 + spacing + cell_size_x/2 + i * (cell_size_x + spacing)
                    y = box_y - dims['width']/2 + spacing + cell_size_y/2 + j * (cell_size_y + spacing)
                    
                    stacking_positions.append([x, y, z])
        
        config['computed_stacking_positions'] = stacking_positions
        
        return stacking_positions


def demo_box_usage():
    """演示如何使用箱子工厂"""
    
    from omni.isaac.core import World
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    
    world = World(stage_units_in_meters=1.0)
    assets_root = get_assets_root_path()
    
    print("\n" + "=" * 60)
    print("码垛箱子管理演示")
    print("=" * 60)
    
    factory = BoxFactory(world, assets_root)
    
    factory.config_loader.list_available_boxes()
    
    print("\n" + "=" * 60)
    print("场景1: 创建小型纸箱")
    print("=" * 60)
    
    box1_path, box1_config = factory.create_box("cardboard_box_small", "box_A")
    
    print("\n" + "=" * 60)
    print("场景2: 创建多个不同箱子")
    print("=" * 60)
    
    box2_path, box2_config = factory.create_box("plastic_crate", "box_B", custom_position=[1.5, 0.5, 0.0])
    box3_path, box3_config = factory.create_box("wooden_pallet_box", "box_C", custom_position=[3.0, 0.5, 0.0])
    
    print("\n" + "=" * 60)
    print("场景3: 访问码垛位置")
    print("=" * 60)
    
    positions = box1_config['computed_stacking_positions']
    print(f"\n箱子A的前5个码垛位置:")
    for idx, pos in enumerate(positions[:5], 1):
        print(f"  位置{idx}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print("\n" + "=" * 60)
    print("所有箱子已就绪，可以开始码垛!")
    print("=" * 60)


if __name__ == "__main__":
    demo_box_usage()