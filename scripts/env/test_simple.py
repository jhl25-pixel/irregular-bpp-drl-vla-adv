# ============================================================
# 文件: test_simple.py
# ============================================================

from isaacsim import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "headless": True,
    "hide_ui": True,
    "renderer": "RaytracedLighting",
}

print("=" * 60)
print("步骤1: 初始化 Isaac Sim (Headless模式)")
print("=" * 60)

simulation_app = SimulationApp(CONFIG)

print("\n" + "=" * 60)
print("步骤2: Isaac Sim 启动成功!")
print("=" * 60)

print("\n===== 导入Isaac模块")
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path

print("✓ Isaac模块导入成功")

print("\n===== 创建World")
world = World(stage_units_in_meters=1.0)
assets_root = get_assets_root_path()

print(f"✓ World创建成功")
print(f"✓ Assets路径: {assets_root}")

print("\n===== 重置World")
world.reset()
print("✓ World重置成功")

print("\n===== 运行10帧测试")
for i in range(10):
    world.step(render=True)
    print(f"  帧 {i+1}/10")

print("\n" + "=" * 60)
print("✓ 所有测试完成!")
print("=" * 60)

simulation_app.close()