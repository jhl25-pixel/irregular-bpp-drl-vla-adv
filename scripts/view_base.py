from isaacsim import SimulationApp

# 配置仿真应用
CONFIG = {
    "headless": False,        # False=显示界面, True=无界面运行
    "width": 1280,
    "height": 720,
}
simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import World

# 创建世界（场景的容器）
world = World(stage_units_in_meters=1.0)

# 添加地面
world.scene.add_default_ground_plane()

# 重置世界
world.reset()

# 运行仿真循环
while simulation_app.is_running():
    world.step(render=True)  # 每次迭代更新一步

simulation_app.close()