import numpy as np
import argparse
import random
from collections import defaultdict
from isaacsim import SimulationApp
import time

# 1. argparse 설정
parser = argparse.ArgumentParser(description="Isaac Sim Dynamic Stabilization Generator")
parser.add_argument("--target_name", type=str, required=True, help="Target USD name")
parser.add_argument("--num_images", type=int, default=1000, help="Number of scenes")
args, unknown = parser.parse_known_args()
simulation_app = SimulationApp({"headless": False})

import omni.usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.viewports import create_viewport_for_camera, destroy_all_viewports, set_active_viewport_camera
from pxr import Sdf, UsdLux
from isaacsim.core.utils.stage import add_reference_to_stage

from omni.isaac.cloner import Cloner
import omni.isaac.core.utils.prims as prim_utils

from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport_window
# --- 데이터 경로 로드 ---
MUG = {f"mug_{i}": f"/home/irol/Desktop/MH_FILE/mug/mug_{i}/mug_{i}.usd" for i in range(1, 5)}
CAN = {f"can_{i}": f"/home/irol/Desktop/MH_FILE/can/can_{i}/can_{i}.usd" for i in range(1, 4)}
SNACK = {f"snack_{i}": f"/home/irol/Desktop/MH_FILE/snack/snack_{i}/snack_{i}.usd" for i in range(1, 4)}
SCREW = {f"screw_{i}": f"/home/irol/Desktop/MH_FILE/screw/screw_{i}/screw_{i}.usd" for i in range(1, 5)}

SIMILARITY_MAP = {
    'mug': {'mug': 0.8, 'can': 0.5, 'snack': 0.2, 'screw':0.2},
    'can': {'mug': 0.5, 'can': 0.8, 'snack': 0.2, 'screw':0.2},
    'snack': {'mug': 0.2, 'can': 0.2, 'snack': 0.8, 'screw':0.2},
    'screw' : {'mug': 0.2, 'can': 0.2, 'snack': 0.2, 'screw':0.8}
}

SCORE_TO_RADIUS = {0.8: 0.03, 0.5: 0.06, 0.2: 0.09}
WORKSPACE_BOUNDS = {'x': [-0.25, 0.25], 'y': [-0.25, 0.25], 'z': [0.2, 0.35]}

class Cluttered_Scene_Generator:
    def __init__(self):
        self.world = World(physics_dt=1/60.0)
        self.items = {}
        self.item_types = {}
        self.wait_positions = {}
        self.last_orientations = {} # 이전 프레임의 회전값 저장용
        self._scene_initialize()

    def _scene_initialize(self):
        GroundPlane(prim_path="/World/GroundPlane", z_position=0, color=np.array([1.0, 1.0, 1.0]))
        distantLight = UsdLux.DistantLight.Define(self.world.stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(2000)



        self.camera_01 = Camera(prim_path="/World/camera01", position=np.array([0.0, 0.0, 3.5]),
                             resolution=(640, 480), orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 90]), degrees=True))

        self.camera_02 = Camera(prim_path="/World/camera02", position=np.array([0.0, 0.0, 3.5]),
                             resolution=(640, 480), orientation=rot_utils.euler_angles_to_quats(np.array([45, 90, 90]), degrees=True))
        
        self.camera_01.initialize()
        self.camera_02.initialize()

        self.camera_01.add_distance_to_camera_to_frame()
        self.camera_02.add_distance_to_camera_to_frame()

        destroy_all_viewports()

        # self.viewport_win_01 = create_viewport_for_camera(viewport_name="Camera Viewport 01", camera_prim_path=self.camera_01.prim_path)
        # set_active_viewport_camera(camera_prim_path=self.camera_01.prim_path)

        # self.viewport_win_02 = create_viewport_for_camera(viewport_name="Camera Viewport 02", camera_prim_path=self.camera_02.prim_path)
        # set_active_viewport_camera(camera_prim_path=self.camera_02.prim_path)

        workspace_usd_path = "/home/irol/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/src/USD_FILE/workspace.usd"
        workspace_prim_path = "/World/workspace"
        add_reference_to_stage(usd_path=workspace_usd_path, prim_path=workspace_prim_path)
        self.workspace_prim = SingleGeometryPrim(prim_path="/World/workspace", visible=True, collision=True)
        self.world.scene.add(self.workspace_prim)

        def load_items(item_dict, type_name, y_offset):
            for i, (key, value) in enumerate(item_dict.items()):
                item_prim_path = f"/World/Item/{key}"
                add_reference_to_stage(usd_path=value, prim_path=item_prim_path)
                wait_pos = np.array([0.8, y_offset, 0.1 * i]) # 서랍 밖 대기 위치
                item_rigid_prim = SingleRigidPrim(prim_path=item_prim_path, name=key, position=wait_pos)
                self.items[key] = item_rigid_prim
                self.item_types[key] = type_name
                self.wait_positions[key] = wait_pos
                self.world.scene.add(item_rigid_prim)

        load_items(MUG, "mug", 0.0)
        load_items(CAN, "can", 0.2)
        load_items(SNACK, "snack", 0.45)
        load_items(SCREW, "screw", 0.65)
        self.world.reset()

    def check_stabilization(self, threshold=1e-4):
        """
        [추가] 서랍 안에 있는 모든 물체의 회전 변화량을 체크하여 안정화 여부 판단
        """
        all_stable = True
        for key, obj in self.items.items():
            pos, quat = obj.get_world_pose()
            
            # 서랍 밖 대기 영역에 있는 물체는 체크 제외 (x > 0.5)
            if pos[0] > 0.5: continue 
            
            if key in self.last_orientations:
                # 쿼터니언 내적을 통한 회전 차이 계산 (1.0에 가까울수록 변화 없음)
                dot_product = np.abs(np.dot(quat, self.last_orientations[key]))
                diff = 1.0 - dot_product
                
                # 속도와 회전 변화량이 모두 낮아야 안정화로 판단
                if diff > threshold:
                    all_stable = False
            
            # 현재 회전값 업데이트
            self.last_orientations[key] = quat
            
        return all_stable

    def wait_for_stabilization(self, max_steps=500):
        """안정화될 때까지 대기 (최대 max_steps)"""
        for _ in range(max_steps):
            self.world.step(render=True)
            if self.check_stabilization():
                break

    def reset_items(self):
        for key, obj in self.items.items():
            obj.set_world_pose(position=self.wait_positions[key])
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        self.last_orientations = {} # 회전 기록 초기화
        for _ in range(10): self.world.step()

    def generate_one_scene(self, idx):
        target_name = args.target_name   
        print(f"--- Scene {idx+1}: Target {target_name} ---")
        self.reset_items()
        
        # 1. Target 배치 및 안정화
        target_obj = self.items[target_name]
        target_pos = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.2])
        target_obj.set_world_pose(position=target_pos)
        self.wait_for_stabilization(max_steps=120)

        # 2. 주변 물체 순차 투하
        target_type = self.item_types[target_name]
        score_groups = defaultdict(list)
        for k, v in self.item_types.items():
            if k == target_name: continue
            score = SIMILARITY_MAP[target_type][v]
            score_groups[score].append(k)

        for score in sorted(score_groups.keys(), reverse=True):
            group = score_groups[score]
            random.shuffle(group)
            for item_key in group:
                obj = self.items[item_key]
                radius = SCORE_TO_RADIUS.get(score, 0.2)
                
                # 가우시안 대신 원형 범위 내 스폰 로직
                r = radius * np.sqrt(random.random())
                theta = random.random() * 2 * np.pi
                cur_t_pos, _ = target_obj.get_world_pose()
                spawn_pos = np.array([
                    np.clip(cur_t_pos[0] + r*np.cos(theta), -0.3, 0.3),
                    np.clip(cur_t_pos[1] + r*np.sin(theta), -0.3, 0.3),
                    random.uniform(0.3, 0.5)
                ])
                
                obj.set_world_pose(position=spawn_pos)
                # 물체 하나 투하할 때마다 최소한의 물리 안정화
                self.wait_for_stabilization(max_steps=60)

        # 3. 모든 물체 투하 후 최종 안정화
        print("Final stabilizing...")
        self.wait_for_stabilization(max_steps=300)

        img_1 = self.camera_01.get_current_frame()['distance_to_camera']
        img_2 = self.camera_02.get_current_frame()['distance_to_camera']

        np.save(f"/home/irol/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/src/test/scene_{idx+1:04d}_cam1.npy", img_1)
        np.save(f"/home/irol/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/src/test/scene_{idx+1:04d}_cam2.npy", img_2)

        # capture_viewport_to_file(self.viewport_win_01.viewport_api, f"/home/irol/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/src/test/scene_{idx+1:04d}_cam1.png")
        # capture_viewport_to_file(self.viewport_win_02.viewport_api, f"/home/irol/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/src/test/scene_{idx+1:04d}_cam2.png")
        print(f"Scene {idx+1} Stabilized.")
        
def main():
    generator = Cluttered_Scene_Generator()
    for i in range(args.num_images):
        generator.generate_one_scene(i)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", type(e))
    finally:
        simulation_app.close()
