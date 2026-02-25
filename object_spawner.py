"""ObjectSpawner — reusable class for spawning USD objects onto a workspace."""

from __future__ import annotations

import os
import random
from typing import Optional

import torch

from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrimView, XFormPrimView


class ObjectSpawner:
    """Scans one or more USD category directories and spawns objects as
    rigid-body children of a single XForm container.

    Lifecycle
    ---------
    1. ``__init__``  – discovers assets across all *categories*, creates
       the XForm container, and loads all prims at a **default position
       outside** the workspace (staging area).
    2. ``spawn()``   – moves the prims **into** the workspace bounds
       (random or ordered placement).
    3. ``initialize()`` – moves all prims **back** to the default
       position (outside the workspace).  No prims are removed.

    Parameters
    ----------
    world : isaacsim.core.api.World
        The simulation world instance.
    categories : str | list[str]
        One or more category names.  Each must be a subdirectory of
        *usd_folder_dir* (e.g. ``["Food", "Toy"]``).
    usd_folder_dir : str
        Parent directory that contains per-category subdirectories.
    container_prim_path : str
        Stage path for the XForm container
        (e.g. ``"/World/Objects_0"``).
    workspace_bounds : dict
        Spawn area on the workspace surface.
        ``{"x": (min, max), "y": (min, max), "z_surface": float}``
    default_position : torch.Tensor | None
        Position outside the workspace where prims are parked
        when not in use.  Defaults to ``[10, 10, -1]``.
    num_to_spawn : int | None
        How many objects to spawn.  ``None`` → spawn all found assets.
    extensions : tuple[str, ...]
        File extensions to treat as valid USD assets.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        world,
        categories: str | list[str],
        usd_folder_dir: str,
        container_prim_path: str,
        workspace_bounds: dict,
        default_position: Optional[torch.Tensor] = None,
        num_to_spawn: Optional[int] = None,
        extensions: tuple[str, ...] = (".usd", ".usdc"),
    ) -> None:
        self._world = world
        self._categories = (
            [categories] if isinstance(categories, str) else list(categories)
        )
        self._usd_folder_dir = usd_folder_dir
        self._container_path = container_prim_path
        self._bounds = workspace_bounds
        self._extensions = extensions
        self._device = self._world.device
        self._default_position = (
            default_position.to(self._device) if default_position is not None
            else torch.tensor([10.0, 10.0, -1.0], device=self._device)
        )

        # Extract the container prefix from the prim path
        # e.g. "/World/Objects_0" → "Objects"
        self._container_prefix = (
            container_prim_path.rsplit("/", 1)[-1].rsplit("_", 1)[0]
        )

        # Discover assets across all categories
        # objects_dir:   {"cracker_box": "/…/Food/cracker_box.usd", …}
        # objects_class:  {"cracker_box": "Food", "teddy_bear": "Toy", …}
        self._objects_dir: dict[str, str] = {}
        self._objects_class: dict[str, str] = {}

        for category in self._categories:
            usd_dir = os.path.join(self._usd_folder_dir, category)
            if not os.path.isdir(usd_dir):
                raise FileNotFoundError(
                    f"Category directory not found: {usd_dir}"
                )
            for f in os.listdir(usd_dir):
                if f.endswith(extensions):
                    name = os.path.splitext(f)[0]
                    self._objects_dir[name] = os.path.join(usd_dir, f)
                    self._objects_class[name] = category

        if not self._objects_dir:
            raise FileNotFoundError(
                f"No USD assets ({extensions}) found in categories "
                f"{self._categories} under {usd_folder_dir}"
            )

        self._num_to_spawn = num_to_spawn or len(self._objects_dir)
        self._spawned_prims: list[SingleRigidPrim] = []
        self._spawned_paths: list[str] = []
        self._spawned_names: list[str] = []  # object names in spawn order

        # Set after setup_cloned_views() is called
        self._num_envs: int = 1
        self._item_views: list[RigidPrimView] = []
        self._container_view: XFormPrimView | None = None

        # Create the container and load all prims at the default position
        self._create_prims()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup_cloned_views(self, num_envs: int) -> list[RigidPrimView]:
        """Create per-item RigidPrimViews that span all cloned envs.

        Must be called **after** ``GridCloner.clone()``.

        Parameters
        ----------
        num_envs : int
            Total number of cloned environments.

        Returns
        -------
        list[RigidPrimView]
            One view per item, each of shape (num_envs, …).
        """
        self._num_envs = num_envs
        self._item_views.clear()

        # Container view for world-position lookups
        self._container_view = XFormPrimView(
            f"/World/{self._container_prefix}_*"
        )

        for i in range(self._num_to_spawn):
            pattern = (
                f"/World/{self._container_prefix}_*/object_{i}"
            )
            view = RigidPrimView(pattern)
            self._item_views.append(view)

        return self._item_views

    def spawn(self, randomize: bool = True) -> None:
        """Move items into workspace bounds across all cloned envs.

        Computes **world-space** positions by adding local offsets to
        each container's world position, then uses
        ``RigidPrimView.set_world_poses()`` to properly teleport the
        rigid bodies.

        Parameters
        ----------
        randomize : bool
            If True, each environment gets independent random (x, y).
            If False, evenly-spaced placement (same across all envs).
        """
        container_world_pos, _ = self._container_view.get_world_poses()

        for i, view in enumerate(self._item_views):
            local_offsets = torch.stack([
                self._compute_workspace_position(i, randomize)
                for _ in range(self._num_envs)
            ])
            world_positions = container_world_pos + local_offsets
            view.set_world_poses(world_positions)

    def initialize(self) -> None:
        """Move all items back to the default position (outside the
        workspace) across all cloned environments."""
        container_world_pos, _ = self._container_view.get_world_poses()
        default_offset = self._default_position.unsqueeze(0).expand(
            self._num_envs, -1
        )
        world_positions = container_world_pos + default_offset
        for view in self._item_views:
            view.set_world_poses(world_positions)

    def get_prim_paths(self) -> list[str]:
        """Return prim paths of currently managed objects."""
        return list(self._spawned_paths)

    @property
    def container_path(self) -> str:
        return self._container_path

    @property
    def num_spawned(self) -> int:
        return len(self._spawned_prims)

    @property
    def default_position(self) -> torch.Tensor:
        return self._default_position

    @property
    def objects_dir(self) -> dict[str, str]:
        """Mapping of object name → USD file path."""
        return dict(self._objects_dir)

    @property
    def objects_class(self) -> dict[str, str]:
        """Mapping of object name → category name."""
        return dict(self._objects_class)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_prims(self) -> None:
        """Create the XForm container and load all asset prims at the
        default (outside-workspace) position.  Called once in __init__."""
        stage = self._world.scene.stage

        # Create or recreate the container XForm
        if stage.GetPrimAtPath(self._container_path):
            stage.RemovePrim(self._container_path)
        SingleXFormPrim(
            prim_path=self._container_path,
            name=f"{self._container_prefix.lower()}_container",
        )

        # Load assets (up to num_to_spawn)
        asset_items = list(self._objects_dir.items())[: self._num_to_spawn]

        for i, (name, asset_path) in enumerate(asset_items):
            prim_path = f"{self._container_path}/object_{i}"

            add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)

            prim = SingleRigidPrim(
                prim_path=prim_path,
                name=f"{self._container_prefix}_{name}",
                position=self._default_position,
            )
            prim.set_default_state(position=self._default_position)
            self._world.scene.add(prim)

            self._spawned_prims.append(prim)
            self._spawned_paths.append(prim_path)
            self._spawned_names.append(name)

    def _compute_workspace_position(
        self, index: int, randomize: bool
    ) -> torch.Tensor:
        """Compute a position on the workspace surface."""
        x_min, x_max = self._bounds["x"]
        y_min, y_max = self._bounds["y"]
        z = self._bounds["z_surface"]

        if randomize:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
        else:
            t = index / max(self._num_to_spawn - 1, 1)
            x = x_min + t * (x_max - x_min)
            y = (y_min + y_max) / 2.0

        return torch.tensor([x, y, z], device=self._device)
