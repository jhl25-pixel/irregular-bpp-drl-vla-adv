# ============================================================
# 文件: scripts/env/environment_loader.py
# ============================================================

import yaml
import numpy as np
from pathlib import Path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import UsdGeom, Gf
import omni.usd


class EnvironmentConfigLoader:
    """环境配置加载器"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_dir = project_root / "config" / "environment_configs"
        
        self.config_dir = Path(config_dir)
        self.configs = {}
        
        print("=" * 60)
        print("初始化环境配置加载器")
        print(f"配置目录: {self.config_dir}")
        print(f"目录存在: {self.config_dir.exists()}")
        print("=" * 60)
        
        if self.config_dir.exists():
            self._load_all_configs()
    
    def _load_all_configs(self):
        """加载所有环境配置"""
        yaml_files = list(self.config_dir.glob("*.yaml"))
        
        print(f"\n发现 {len(yaml_files)} 个环境配置文件:")
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    env_name = config_data['environment_config']['scene_name']
                    self.configs[env_name] = config_data['environment_config']
                    print(f"  ✓ 加载配置: {env_name} ({yaml_file.name})")
            except Exception as e:
                print(f"  ✗ 加载失败: {yaml_file.name} - {e}")
        
        print("=" * 60)
    
    def get_config(self, env_name):
        """获取环境配置"""
        return self.configs.get(env_name)
    
    def list_available_environments(self):
        """列出可用环境"""
        print("\n" + "=" * 60)
        print("可用环境列表:")
        print("=" * 60)
        
        for i, (name, config) in enumerate(self.configs.items(), 1):
            print(f"\n{i}. {name}")
            print(f"   类型: {config.get('scene_type', 'N/A')}")
            if config.get('base_scene', {}).get('use_builtin'):
                print(f"   场景: Isaac Sim 预制场景")
                print(f"   路径: {config['base_scene']['builtin_path']}")


class EnvironmentFactory:
    """环境工厂 - 加载预制场景并集成自定义物体"""
    
    def __init__(self, world, assets_root):
        self.world = world
        self.assets_root = assets_root
        self.config_loader = EnvironmentConfigLoader()
        self.loaded_scene = None
        
        print("\n" + "=" * 60)
        print("环境工厂初始化完成")
        print("=" * 60)
    
    def load_environment(self, env_name):
        """加载环境"""
        print("\n" + "=" * 60)
        print(f"开始加载环境: {env_name}")
        print("=" * 60)
        
        config = self.config_loader.get_config(env_name)
        if not config:
            print(f"✗ 未找到环境配置: {env_name}")
            return None
        
        # 加载基础场景
        if config.get('base_scene', {}).get('use_builtin'):
            scene_info = self._load_builtin_scene(config['base_scene'])
        
        # 返回场景信息和工作区域
        return {
            'scene_prim': scene_info,
            'work_zones': config.get('work_zones', []),
            'settings': config.get('settings', {}),
        }
    
    def _load_builtin_scene(self, scene_config):
        """加载Isaac Sim内置场景"""
        print("\n步骤1: 加载Isaac Sim预制场景")
        
        # 完整的USD路径
        builtin_path = scene_config['builtin_path']
        full_path = self.assets_root + builtin_path
        prim_path = scene_config.get('prim_path', '/World/Environment')
        
        print(f"  场景路径: {full_path}")
        print(f"  Prim路径: {prim_path}")
        
        # 加载场景
        add_reference_to_stage(usd_path=full_path, prim_path=prim_path)
        
        print("  ✓ 场景加载完成")
        
        return prim_path
    
    def get_work_zone_transform(self, zone_name):
        """获取工作区域的变换矩阵（用于放置物体）"""
        if not self.loaded_scene:
            return None
        
        for zone in self.loaded_scene.get('work_zones', []):
            if zone['name'] == zone_name:
                return {
                    'position': np.array(zone['center']),
                    'size': np.array(zone['size']),
                }
        return None


# ============================================================
# 使用示例
# ============================================================

def example_usage():
    """示例：如何使用环境加载器"""
    from omni.isaac.core import World
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    
    # 创建World
    world = World(stage_units_in_meters=1.0)
    assets_root = get_assets_root_path()
    
    # 创建环境工厂
    env_factory = EnvironmentFactory(world, assets_root)
    
    # 列出可用环境
    env_factory.config_loader.list_available_environments()
    
    # 加载工厂环境
    scene_info = env_factory.load_environment("factory_warehouse")
    
    if scene_info:
        print("\n✓ 环境加载成功")
        print(f"工作区域数量: {len(scene_info['work_zones'])}")
        
        # 获取工作区域位置（用于放置机器人）
        zone1 = env_factory.get_work_zone_transform("palletizing_zone_1")
        if zone1:
            print(f"\n工作区域1位置: {zone1['position']}")
            print(f"工作区域1大小: {zone1['size']}")
    
    return env_factory, scene_info