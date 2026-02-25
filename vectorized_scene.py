import os
import argparse
import torch
import numpy as np
from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Isaac Sim Object Spawner with Target Priority")
    
# --target 옵션 추가 (기본값 설정 가능)
# parser.add_argument(
#     "--target", 
#     type=str, 
#     default="food_1", 
#     help="가장 먼저 스폰할 목표 물체의 이름 (예: Pen_2, Eraser_1)"
# )

# args = parser.parse_args()

# ==============================================================================
# 1. Launch Simulation App
# ==============================================================================
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import isaacsim.core.utils.torch.rotations as rot_utils

from object_spawner import ObjectSpawner

# ==============================================================================
# 2. Configuration
# ==============================================================================
NUM_ENVS = 4
GRID_SPACING = 3
CAMERA_HEIGHT_OFFSET = 3.5

SRC_DIR = os.path.dirname(__file__)
USD_FILE_DIR = os.path.join(SRC_DIR, "USD_FILE")

WORKSPACE_USD_PATH = os.path.join(USD_FILE_DIR, "drawer.usd")

# Workspace surface bounds for random object placement
WORKSPACE_BOUNDS = {
    "x": (-0.3, 0.3),
    "y": (-0.2, 0.2),
    "z_surface": 0.5,
}

# ==============================================================================
# 3. Create World & Ground Plane
# ==============================================================================
world = World(physics_dt=1 / 60.0, backend="torch", device="cuda")
GroundPlane(
    prim_path="/World/GroundPlane",
    z_position=0,
    color=torch.tensor([1.0, 1.0, 1.0]),
)

# ==============================================================================
# 4. Build Base Environment (env_0)
# ==============================================================================
# 4-1. Workspace
add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace_0")

# 4-2. Create objects at default position (outside workspace)
object_spawner = ObjectSpawner(
    world=world,
    categories=["Food", "Toy"],
    usd_folder_dir=USD_FILE_DIR,
    container_prim_path="/World/Objects_0",
    workspace_bounds=WORKSPACE_BOUNDS,
    default_position=torch.tensor([0.0, 0.7, 0.05]),
    num_to_spawn=None,          # load all available assets
    extensions=(".usd",),       # only .usd files
)

# 4-3. Camera
camera = Camera(
    prim_path="/World/camera_0",
    position=torch.tensor([0.0, 0.0, 0.0]),
    resolution=(640, 480),
    orientation=rot_utils.euler_angles_to_quats(
        torch.tensor([0, 0, 0]), degrees=True
    ),
)
camera.initialize()

# ==============================================================================
# 5. Clone Environments
# ==============================================================================
cloner = GridCloner(spacing=GRID_SPACING)

workspace_paths = cloner.generate_paths("/World/workspace", NUM_ENVS)
object_paths = cloner.generate_paths("/World/Objects", NUM_ENVS)
camera_paths = cloner.generate_paths("/World/camera", NUM_ENVS)

cloner.clone(source_prim_path="/World/workspace_0", prim_paths=workspace_paths)
cloner.clone(source_prim_path="/World/Objects_0", prim_paths=object_paths)
cloner.clone(source_prim_path="/World/camera_0", prim_paths=camera_paths)

# ==============================================================================
# 6. Arrange Cloned Poses
# ==============================================================================
workspaces_view = XFormPrimView("/World/workspace_*")
objects_view = XFormPrimView("/World/Objects_*")
cameras_view = XFormPrimView("/World/camera_*")

# Align object containers with workspace positions
positions, orientations = workspaces_view.get_world_poses()
workspaces_view.set_world_poses(positions, orientations)
objects_view.set_world_poses(positions, orientations)

# Offset cameras above the workspaces
cam_positions, cam_orientations = cameras_view.get_world_poses()
cam_positions[:, 2] += CAMERA_HEIGHT_OFFSET
cameras_view.set_world_poses(cam_positions, cam_orientations)

# Create per-item views that span all cloned environments
object_spawner.setup_cloned_views(num_envs=NUM_ENVS)

# ==============================================================================
# 7. Simulation Loop
# ==============================================================================
world.play()

# Start with objects at default (outside workspace)
object_spawner.initialize()
world.step(render=True)

# Spawn objects into workspaces
object_spawner.spawn(randomize=True)

count = 0
while simulation_app.is_running():
    if count % 100 == 0:
        object_spawner.spawn()
    world.step(render=True)
    count += 1