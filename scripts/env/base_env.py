# isaacsim_related
from isaacsim import SimulationApp
CONFIG = {
    "width": 1707,
    "height": 1280,
    "window_width": 3840,
    "window_height": 2160,
    "headless": True,
    "hide_ui": False,  # Show the GUI   
    "renderer": "RaytracedLighting",
    "display_options": 3286,  # Set display options to show default grid
}
simulation_app = SimulationApp(CONFIG)
from isaacsim.core.utils.extensions import enable_extension

# Default Livestream settings
simulation_app.set_setting("/app/window/drawMouse", False)

# Enable Livestream extension
enable_extension("omni.kit.livestream.webrtc")

# core
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdLux

# isaacsim/franka
from omni.isaac.core.utils.nucleus import get_assets_root_path
# isaacsim/object
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
# isaacsim/load_pcd
import omni.usd
from pxr import UsdGeom, Vt, Gf, UsdShade,Sdf
import omni.replicator.core as rep
# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
# cuRobo/motion_generation
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# third_party
import numpy as np
import cv2
import open3d as o3d
import torch
import sys
import argparse
import os
import carb

# user_party
try:
    from .utlis import *
except:
    from utlis import *

# args
from types import SimpleNamespace
import yaml

from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
# from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
try:
    from .sim_controller import controller as RMPFlowController
except:
    from sim_controller import controller as RMPFlowController
from isaacsim.core.utils.types import ArticulationAction

import sys
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.kit.viewport.utility as vp_utils  
from pxr import Usd, UsdPhysics
from omni.isaac.franka import Franka
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid

# 获取Python解释器的绝对路径
python_path = sys.executable

print(f"Python解释器路径: {python_path}")
def parse_args(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, 
                        help='Path to the config file')
    cmd_args = parser.parse_args()
    with open(cmd_args.config, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    args_dict = vars(cmd_args)  # 将args转换为字典
    args_dict.update(yaml_config)  # 用YAML配置更新字典
    args = SimpleNamespace(**args_dict)
    return args

class Base_Env():
    def __init__(self,args):
        # cal
        self.cal = TfCal()
        self.args = args
        self.world = World(stage_units_in_meters=1.0)
        self.assets_root = get_assets_root_path()
        # self.my_franka = self.load_robot()
        self.my_franka = self.load_new_robot()
        self.my_fr3 = self.load_robot()
        self.load_camera()
        self.load_ground()

        # controller_related
        self.world.reset()
        
        self.controller = RMPFlowController(robot_articulation=self.my_franka)
        self.articulation_controller = self.my_franka.get_articulation_controller()
        
        #scene related
        if args.modify_light:
            self.modify_lighting()

    # math_tools
    def usd_euler_to_quat(self, euler, degrees=False):
        """
        使用USD的Gf.Rotation类实现欧拉角到四元数的转换
        完全依赖USD库函数，无自定义转换逻辑
        """
        roll = float(euler[0])
        pitch = float(euler[1])
        yaw = float(euler[2])
    
        if degrees:
            # 转换为弧度
            roll = np.radians(roll)
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)
        
        # 1. 创建三个轴的旋转
        rot_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), roll)  # 绕x轴旋转(roll)
        rot_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch) # 绕y轴旋转(pitch)
        rot_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)   # 绕z轴旋转(yaw)
        
        # 2. 组合旋转（顺序：roll→pitch→yaw）
        total_rot = rot_z * rot_y * rot_x  # 注意乘法顺序：先应用x旋转，再y，最后z
        
        # 3. 转换为四元数
        quat = total_rot.GetQuat()
        
        # 4. 提取为[w, x, y, z]格式
        return [quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]]
    
    # load_related   
    def add_obj_object(self,obj_file):
        add_reference_to_stage(usd_path=obj_file, prim_path="/World/MyOBJ")#TODO add pose and scale 
        #/home/thummlab/arlen/isaac/control_your_robot/tools/convert/usd_convert.py

    # def add_usd_object(self,usd_file,translation=None, orintation=[1.0, 0.0, 0.0, 0.0],scale=[1,1,1],name="MyUSDModel"):
    #     stage = omni.usd.get_context().get_stage()
    #     usd_file_path = usd_file
    #     prim_path = f"/World/{name}"
        
    #     static_friction=10.0  # 静摩擦系数（默认设为 10，通常 0.1~10 是合理范围）
    #     dynamic_friction=10.0  # 动摩擦系数
                
    #     # 定义 Xform prim 并引用 USD 文件
    #     prim = stage.DefinePrim(prim_path, "Xform")
    #     prim.GetReferences().AddReference(usd_file_path)

    #     # 启用物理属性（刚体和碰撞）
    #     UsdPhysics.RigidBodyAPI.Apply(prim)
    #     rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
    #     rigid_api.CreateRigidBodyEnabledAttr(True)
    #     collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    #     collision_api.CreateCollisionEnabledAttr(True)

    #     # 设置高摩擦系数
    #     material_api = UsdPhysics.MaterialAPI.Apply(prim)
    #     material_api.CreateStaticFrictionAttr().Set(static_friction)  # 静摩擦
    #     material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)  # 动摩擦

    #     # # 可视化设置（可选）
    #     # prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
    #     # prim.CreateAttribute("render:visibility", Sdf.ValueTypeNames.Token).Set("invisible")
    #     # prim.CreateAttribute("geom:visibility", Sdf.ValueTypeNames.Token).Set("invisible")

    #     # 设置变换（位置、旋转、缩放）
    #     xform = UsdGeom.Xformable(prim)
    #     for op in xform.GetOrderedXformOps():
    #         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
    #             op.Set(Gf.Vec3d(translation.tolist()))  # 位置
    #         elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
    #             op.Set(Gf.Quatf(*orintation))  # 旋转
    #         elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
    #             op.Set(Gf.Vec3f(scale[0], scale[1], scale[2]))  # 缩放

    def add_usd_object(self,usd_file,translation=None,orintation=[1.0, 0.0, 0.0, 0.0],scale=[1,1,1],name="MyUSDModel"):
        stage = omni.usd.get_context().get_stage()

        usd_file_path = usd_file 
        prim_path = f"/World/{name}"

        # 定义一个 Xform prim 并引用 USD 文件
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(usd_file_path)
        
        UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        rigid_api.CreateRigidBodyEnabledAttr(True)  # 启用刚体
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(True)
        
        # # sf.ValueTypeNames.Token).Set("invisible")
  
        # VIS可视化设置（可选）
        prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
        prim.CreateAttribute("render:visibility", Sdf.ValueTypeNames.Token).Set("invisible")
        prim.CreateAttribute("geom:visibility", Sdf.ValueTypeNames.Token).Set("invisible")
  
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(translation.tolist()))  # 如果原来是 double3
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatf(*orintation))  # 如果原来是 quatf
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(Gf.Vec3f(scale[0], scale[1], scale[2]))  # 如果原来是 float3
    
    def remove_usd_object(self, name):
        """
        删除指定名称的USD物体
        
        Args:
            name (str): 要删除的物体名称
        """
        try:
            stage = omni.usd.get_context().get_stage()
            prim_path = f"/World/{name}"
            
            # 检查物体是否存在
            if stage.GetPrimAtPath(prim_path):
                # 删除物体
                stage.RemovePrim(prim_path)
                print(f"成功删除物体: {name}")
                return True
            else:
                print(f"物体不存在: {name}")
                return False
                
        except Exception as e:
            print(f"删除物体时发生错误: {e}")
            return False
    def add_cube(self,position=[0.5, 0.5, 0.5],scale=np.array([0.01, 0.01, 0.01]),mode="static",name=None,orientation = None):#static不可移动,dynamic可以移动
        if mode == "static":
            self.visual_cube = VisualCuboid(
                prim_path=f"/World/{mode}_cube_{name}",
                name=f"{name}_cube",
                position=np.array(position),
                orientation = orientation,
                scale=scale,
                size= 1,
                color=np.array([255, 255, 0]),
            ) 

        else:
            self.dynamic_cube = DynamicCuboid(
                prim_path=f"/World/{mode}_cube_{name}",
                name=f"{name}_cube",
                position=np.array(position),
                orientation = orientation,
                scale=scale,
                size = 1,
                color=np.array([0, 255, 255]),
            )
            
            # stage = omni.usd.get_context().get_stage()
            # prim = stage.GetPrimAtPath("/World/dynamic_cube_m")
            # rigid_body = UsdPhysics.RigidBodyAPI(prim)
            # rigid_body.CreateKinematicEnabledAttr(True)
            
    def load_robot(self,translation:np.ndarray=np.array([-5,0,0])):
            asset_path = self.assets_root + "/Isaac/Robots/Franka/FR3/fr3.usd"
            add_reference_to_stage(usd_path=asset_path, prim_path="/World/fr3")

            gripper = ParallelGripper(
                end_effector_prim_path="/World/fr3/fr3_hand",
                joint_prim_names=["fr3_finger_joint1", "fr3_finger_joint2"],
                joint_opened_positions=np.array([0.05, 0.05]),
                joint_closed_positions=np.array([0.0, 0.0]),
                action_deltas=np.array([0.05, 0.05]),
            )

            self.arm = self.world.scene.add(
                SingleManipulator(
                    prim_path="/World/fr3",
                    name="my_fr3",
                    end_effector_prim_name="fr3_rightfinger",
                    gripper=gripper,
                )
            )
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/fr3")
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            for op in ops:
                if op.GetOpName() == "xformOp:translate":
                    op.Set(Gf.Vec3d(self.args.root_postion))
                elif op.GetOpName() == "xformOp:orient":
                    quat_list = self.usd_euler_to_quat(self.args.root_orientation)  # [w, x, y, z]
                    quat = Gf.Quatd(quat_list[0], quat_list[1], quat_list[2], quat_list[3])
                    op.Set(quat)
            self.pre_joint = None
            return self.arm


    def load_new_robot(self):
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.world.scene.object_exists(x)
        )
        robot = Franka(
            prim_path=franka_prim_path, name=franka_robot_name, end_effector_prim_name="panda_hand"
        )
        self.world.scene.add(robot)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/Franka")
        xformable = UsdGeom.Xformable(prim)
        
        print(self.args.root_postion)
        print(self.args.root_orientation)

        # 获取已存在的 translate op
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpName() == "xformOp:translate":
                op.Set(Gf.Vec3d(self.args.panda_root_postion))
            elif op.GetOpName() == "xformOp:orient":
                quat_list = self.usd_euler_to_quat(self.args.panda_root_orientation)  # [w, x, y, z]
                quat = Gf.Quatd(quat_list[0], quat_list[1], quat_list[2], quat_list[3])
                op.Set(quat)
                # op.Set(Gf.Quatd(self.usd_euler_to_quat(self.args.root_orientation)))
        return robot
        
    def load_camera(self):
        # self.camera = Camera(
        #     prim_path="/World/camera",
        #     position = self.args.position,
        #     frequency=15,
        #     resolution=(1707,1280),
        #     orientation = rotate_quaternion(self.args.rotation)#xyz
        # )
        # self.camera.initialize() 
        # stage = omni.usd.get_context().get_stage()
        
        # camera_prim = stage.GetPrimAtPath("/World/camera")
        # camera_prim.GetAttribute("focalLength").Set(17)#18.97
        # camera_prim.GetAttribute("focusDistance").Set(400.0)
        # camera_prim.GetAttribute("horizontalAperture").Set(20.955)
        # camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.1, 20.0))
        # # camera_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.0, 1.0, 0.0, 0.0))
        # # camera_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 1.0))
        # self.intrinsics = self.camera.get_intrinsics_matrix()
        
        # CHANGE
        
        # H, W = 1280, 1707
        # fx = fy = 1545.0
        
        # H, W = 1080, 1920
        # fx = fy = 1081.0
        
        H, W = self.args.H, self.args.W
        fx = fy = self.args.fxy
        # CHANGE
        
        print("Rotation IsaacSIM: ", rotate_quaternion(self.args.rotation))

        self.camera = Camera(
            prim_path="/World/camera",
            position=self.args.position,
            frequency=15,
            resolution=(W, H),   # 注意：这里是 (width, height)
            orientation=rotate_quaternion(self.args.rotation)  # xyz
        )
        self.camera.initialize()
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath("/World/camera")

        # ---- 设置内参匹配 fx=fy=1545 ----
        f = 35.0  # focalLength mm (可调，但 aperture 也要配合)
        aperture_x = f * W / fx
        aperture_y = f * H / fy

        camera_prim.GetAttribute("focalLength").Set(f)
        camera_prim.GetAttribute("horizontalAperture").Set(aperture_x)
        camera_prim.GetAttribute("verticalAperture").Set(aperture_y)

        # 其他参数
        camera_prim.GetAttribute("focusDistance").Set(400.0)
        camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.1, 20.0))

        # 保存 intrinsics
        self.intrinsics = self.camera.get_intrinsics_matrix()

    def load_ground(self):
        if self.args.ground:
            self.world.scene.add_default_ground_plane()
            self.world.scene.add(self.camera)  # 关键：将相机添加到场景中管理
        else:
            pass

    def load_point_cloud(self,point_cloud_file,voxel_size=0.005,voxel_mode=False,point_szie=0.001,tf=False):
        stage = omni.usd.get_context().get_stage()
        points, colors = None, None
        point_cloud_filepath = point_cloud_file
        pcd = o3d.io.read_point_cloud(point_cloud_filepath, print_progress=False)
        if voxel_mode: #决定是否降采样
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        points = np.asarray(pcd.points).astype(np.float32, copy=False)
        colors = np.asarray(pcd.colors).astype(np.float32, copy=False) if pcd.has_colors() else None
        print(f"点云加载完成，共 {len(points)} 个点（降采样后）")

        if tf:
            points = point_cloud_tf(points)
                
        prim_path = "/World/MyPointCloud"
        usd_points = UsdGeom.Points.Define(stage, prim_path)
        
        # 高效设置点数据（批量转换，避免循环）
        usd_points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        usd_points.GetWidthsAttr().Set(Vt.FloatArray([point_szie]))  # 点大小

        # 设置颜色（批量操作）
        color_primvar = usd_points.CreateDisplayColorPrimvar()
        color_primvar.SetInterpolation(UsdGeom.Tokens.vertex)
        color_primvar.Set(Vt.Vec3fArray.FromNumpy(colors))

        print(f"点云已添加到场景: {prim_path}")

    def read_obj_pose_from_txt(self,path):
        pose_path = path
        with open(pose_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        matrix_data = []
        for line in lines:
            numbers = line.strip().split()
            row = [float(num) for num in numbers]
            matrix_data.append(row)
        transform_matrix = np.array(matrix_data)
        return transform_matrix
    
    # visulize_part
    def human_render(self):
        self.world.reset()
        while True:
            self.world.step(render=True)

    def visualize_point_cloud(self,save_pcd=True):
        self.world.reset()  # 初始化所有组件
        self.camera.initialize()  # 显式初始化相机
        for _ in range(10):
            self.world.step(render=True)
        self.camera.add_distance_to_image_plane_to_frame()
        
        target_frame = 50
        current_frame = 0
        while current_frame < target_frame and simulation_app.is_running():
            self.world.step(render=True)
            current_frame += 1        
        
        if simulation_app.is_running() and self.camera._render_product_path is not None:
            current_frame_data = self.camera.get_current_frame()
            depth_image = current_frame_data["distance_to_image_plane"]
        rgb_image = self.camera.get_rgb()  
        # 过滤无效深度值
        depth_min = 0.1
        depth_max = 5.0
        valid_mask = (depth_image > depth_min) & (depth_image < depth_max)

        depth_valid = depth_image[valid_mask]
        rgb_valid = rgb_image[valid_mask]

        # 相机内参
        intrinsics = self.camera.get_intrinsics_matrix()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # 像素坐标
        u, v = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
        u_valid = u[valid_mask].flatten()
        v_valid = v[valid_mask].flatten()

        # 按相机坐标系直接还原3D点（不翻转）
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        points_cam = np.column_stack([x, y, z])

        # 显示点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cam)
        pcd.colors = o3d.utility.Vector3dVector(rgb_valid / 255.0)
        o3d.visualization.draw_geometries([pcd])

        if save_pcd:
                save_dir = "pointcloud_data"
                os.makedirs(save_dir, exist_ok=True)
                ply_path = "pointcloud.ply"
                o3d.io.write_point_cloud(ply_path, pcd)
                carb.log_info(f"点云已保存到: {ply_path}")

    def get_rgb_point_cloud(self):
        self.camera.add_distance_to_image_plane_to_frame() 
        self.world.step(render=True)      
        if simulation_app.is_running() and self.camera._render_product_path is not None :
            self.world.step(render=True)
            current_frame_data = self.camera.get_current_frame()
            depth_image = current_frame_data["distance_to_image_plane"]
            while depth_image is None or len(depth_image)==0:
                self.world.step(render=True)
                current_frame_data = self.camera.get_current_frame()
                depth_image = current_frame_data["distance_to_image_plane"]
        rgb_image = self.camera.get_rgb()  
        # 过滤无效深度值
        depth_min = 0.1
        depth_max = 5.0
        valid_mask = (depth_image > depth_min) & (depth_image < depth_max)
        depth_valid = depth_image[valid_mask]
        rgb_valid = rgb_image[valid_mask]
        
        # 生成点云（相机坐标系）
        intrinsics = self.camera.get_intrinsics_matrix()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # 有效像素坐标
        u, v = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
        u_valid = u[valid_mask].flatten()
        v_valid = v[valid_mask].flatten()
        
        # 转换为3D点
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        points_cam = np.column_stack([x, y, z])
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_cam)
        # pcd.colors = o3d.utility.Vector3dVector(rgb_valid / 255.0)
        return rgb_image, points_cam, rgb_valid,depth_image

    def modify_lighting(self):#TODO set light rigs to defalut
        # 获取当前舞台
        stage = omni.usd.get_context().get_stage()

        # 创建穹顶灯
        dome_prim = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome_light = UsdLux.DomeLight(dome_prim)
        dome_light.CreateIntensityAttr(500.0)  # 光强
        dome_light.CreateColorAttr((1.0, 1.0, 1.0))  # 颜色
        

        # # 聚光灯
        # disk_prim = stage.DefinePrim("/World/DiskLight", "DiskLight")
        # disk_light = UsdLux.DiskLight(disk_prim)
        # disk_light.CreateIntensityAttr(3000.0)
        # disk_light.CreateRadiusAttr(0.2)
        
        # # 矩形灯（常用于区域光）
        # rect_prim = stage.DefinePrim("/World/RectLight", "RectLight")
        # rect_light = UsdLux.RectLight(rect_prim)
        # rect_light.CreateIntensityAttr(500.0)
        # rect_light.CreateWidthAttr(1.0)   # 宽
        # rect_light.CreateHeightAttr(0.5)  # 高
    
    # robot_related
    def movej(self, target_jpos=None, duration=2.0, max_velocity=1.0):#TODO
        """
        使用关节空间插值实现机械臂运动
        
        参数:
        - target_jpos: 目标关节位置 (弧度)
        - duration: 运动持续时间 (秒)
        - max_velocity: 最大关节速度 (弧度/秒)
        """
        
        # 获取当前关节位置
        current_jpos = self.pre_joint
        
        # 计算关节角度差
        target_jpos = np.array(target_jpos)
        current_jpos = np.array(current_jpos)
        joint_delta = target_jpos - current_jpos
        
        # 计算每个关节需要移动的距离
        joint_distances = np.abs(joint_delta)
        
        # 计算最大关节速度下所需的最小时间
        min_time = np.max(joint_distances / max_velocity)
        
        # 确保运动时间不小于用户指定的持续时间
        actual_duration = max(duration, min_time)
        
        # 计算步数 (假设世界步长为1/60秒)
        world_step_size = 1.0 / 60.0  # 默认60 FPS
        num_step = max(1, int(actual_duration / world_step_size))
        
        # 生成等间距的关节位置列表 (线性插值)
        joint_list = []
        for i in range(num_step + 1):  # +1 确保包含终点
            progress = i / num_step
            intermediate_jpos = current_jpos + progress * joint_delta
            joint_list.append(intermediate_jpos)
        
        # 执行运动
        for joint in joint_list:
            self.arm.set_joint_positions(joint.tolist())  # 使用位置目标更稳定
            # for i in range(10):
            self.world.step(render=True)
            

        self.pre_joint = target_jpos

    def move_pose(self,pose,gripper_state="open"):
        while not self.controller.is_done():#is_done from:/home/thummlab/isaacsim/exts/isaacsim.robot_motion.motion_generation/isaacsim/robot_motion/motion_generation/motion_policy_controller.py
            self.world.step(render=True)
            actions = self.controller.forward(
                target_end_effector_position=pose["position"],
                target_end_effector_orientation=pose["orientation"],
            )
            self.articulation_controller.apply_action(actions)
            # self.set_gripper(gripper_state)
        self.controller.iter=0    

    def set_gripper(self,gripper_state):
        if gripper_state == "open":
            gripper_positions = self.my_franka.gripper.get_joint_positions()
            self.my_franka.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + (0.005), gripper_positions[1] + (0.005)])
            )
        elif gripper_state == "close":
            gripper_positions = self.my_franka.gripper.get_joint_positions()
            self.my_franka.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - (0.005), gripper_positions[1] - (0.005)])
            )

    def set_fr3_gripper(self,gripper_state="open                                                                                                           "):
        if gripper_state == "open":
            gripper_positions = self.my_fr3.gripper.get_joint_positions()
            self.my_fr3.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + (0.005), gripper_positions[1] + (0.005)])
            )
        elif gripper_state == "close":
            gripper_positions = self.my_fr3.gripper.get_joint_positions()
            # self.my_fr3.gripper.apply_action(
            #     ArticulationAction(joint_positions=[gripper_positions[0] - (0.005), gripper_positions[1] - (0.005)])
            # )
            self.my_fr3.gripper.apply_action(
                ArticulationAction(
                    joint_positions=[gripper_positions[0] - (0.005), gripper_positions[1] - (0.005)],  # 不指定位置
                    joint_efforts=[0.01, 0.01]  # 给左右两边 gripper 施加力
                )
            )
            
            
    def get_ee_pose(self):
        """用于获取ee相对于机器人坐标系的tf
        """
        try:
            prim = get_prim_at_path("/World/Franka/panda_hand")
            translate = list(prim.GetAttribute("xformOp:translate").Get())
            orient_quat = prim.GetAttribute("xformOp:orient").Get()  # Gf.Quatf
            orient = [orient_quat.GetReal(),
                    orient_quat.GetImaginary()[0],
                    orient_quat.GetImaginary()[1],
                    orient_quat.GetImaginary()[2]]
        except:
            prim = get_prim_at_path("/World/fr3/fr3_hand")
            translate = list(prim.GetAttribute("xformOp:translate").Get())
            orient_quat = prim.GetAttribute("xformOp:orient").Get()  # Gf.Quatf
            orient = [orient_quat.GetReal(),
                    orient_quat.GetImaginary()[0],
                    orient_quat.GetImaginary()[1],
                    orient_quat.GetImaginary()[2]]            

        # quat_list_wxyz = [orient.GetReal()] + list(orient.GetImaginary())
        translate.extend(orient)
        pose = translate
        return np.array(pose)

    def get_ee_translation(self):
        prim = get_prim_at_path("/World/Franka/panda_hand")
        translate = list(prim.GetAttribute("xformOp:translate").Get())
        return translate
    
    def get_joint_position(self):
        return self.my_franka.get_joint_positions()
        
    def get_gripper_move_state(self):
        """
        用于判断夹爪是否完成
        """
        if self.controller.iter == 0:
            self.pre_gripper_positions = self.my_franka.gripper.get_joint_positions()
            self.controller.iter = 1
            self.controller.count+=1
            return False
        if np.allclose(self.my_franka.gripper.get_joint_positions(),self.pre_gripper_positions, atol=1e-4, rtol=1e-4) and self.controller.count>1:
            self.controller.count = 0
            return True
        else:
            self.pre_gripper_positions = self.my_franka.gripper.get_joint_positions()
            self.controller.count += 1
            return False
            
    def _move_gripper(self, step_increment):
        """通用夹爪移动函数
        
        参数:
            step_increment: 步长增量（正数为打开，负数为关闭）
        """
        while not self.get_gripper_move_state():
            self.world.step(render=True)
            current_pos = self.my_franka.gripper.get_joint_positions()
            # 计算新位置（两个夹爪关节同步移动）
            new_pos = [
                current_pos[0] + step_increment,
                current_pos[1] + step_increment
            ]
            self.my_franka.gripper.apply_action(
                ArticulationAction(joint_positions=new_pos)
            )
            # # 应用动作：位置 + 力度
            # self.my_franka.gripper.apply_action(
            #     ArticulationAction(
            #         joint_positions=new_pos,
            #         joint_efforts=[0.001, 0.001]   # 给两个手指相同的力度
            #     )
            # )
        self.controller.iter = 0  # 重置控制器迭代计数

    def open_gripper(self):
        # 打开夹爪（使用正增量）
        self._move_gripper(step_increment=0.005)

    def close_gripper(self):
        # 关闭夹爪（使用负增量）
        self._move_gripper(step_increment=-0.005)

    
if __name__ == "__main__":
    args = parse_args('/home/hljin/irregular-bpp-drl-vla2/config/task_config/base_env.yaml')
    demo = Base_Env(args)
    demo.human_render()


