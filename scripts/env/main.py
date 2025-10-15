import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from isaacsim import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "headless": True,
    "hide_ui": True,
    "renderer": "PathTracing",
    "denoiser": False,
}

print("=" * 60)
print("初始化 Isaac Sim (工厂环境)")
print("=" * 60)

simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path

# 导入你的加载器
from environment_loader import EnvironmentFactory
from robot_loader import RobotFactory
from box_loader import BoxFactory


def main():
    """主程序 - 工厂环境 + 自定义配置"""
    
    print("\n" + "=" * 60)
    print("码垛系统初始化 - 工厂环境版本")
    print("=" * 60)
    
    # 初始化World
    world = World(stage_units_in_meters=1.0)
    assets_root = get_assets_root_path()
    
    # ===== 步骤1: 加载工厂环境 =====
    print("\n[1/3] 加载工厂环境...")
    env_factory = EnvironmentFactory(world, assets_root)
    env_factory.config_loader.list_available_environments()
    
    scene_info = env_factory.load_environment("factory_warehouse")
    
    if not scene_info:
        print("✗ 环境加载失败")
        return
    
    print("✓ 工厂环境加载完成")
    
    # ===== 步骤2: 在工作区域放置机器人 =====
    print("\n[2/3] 在工作区域放置机器人...")
    robot_factory = RobotFactory(world, assets_root)
    
    # 获取工作区域1的位置
    zone1_pos = env_factory.get_work_zone_transform("palletizing_zone_1")
    
    # 在工作区域1创建Franka机器人（覆盖配置中的位置）
    panda_arm, panda_config = robot_factory.create_robot(
        robot_name="franka_panda",
        instance_name="panda_zone1",
        position_override=zone1_pos['position'] if zone1_pos else None
    )
    
    print(f"✓ 机器人放置在工作区域1: {zone1_pos['position'] if zone1_pos else 'default'}")
    
    # ===== 步骤3: 添加箱子（使用你的box配置）=====
    print("\n[3/3] 添加物体...")
    box_factory = BoxFactory(world, assets_root)
    
    # 在工作区域附近放置箱子
    if zone1_pos:
        box_pos = zone1_pos['position'] + np.array([1.0, 0.0, 0.5])
        box_factory.create_box(
            box_name="cardboard_box_small",
            instance_name="box_1",
            position_override=box_pos
        )
    
    print("✓ 物体添加完成")
    
    # ===== 运行仿真 =====
    print("\n" + "=" * 60)
    print("开始仿真")
    print("=" * 60)
    
    world.reset()
    
    for i in range(200):
        world.step(render=True)
        if i % 50 == 0:
            print(f"  仿真步骤: {i}/200")
    
    print("\n" + "=" * 60)
    print("✓ 仿真完成")
    print("=" * 60)
    
    simulation_app.close()


if __name__ == "__main__":
    main()