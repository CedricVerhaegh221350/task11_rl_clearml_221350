"""
Gymnasium-compatible wrapper for the Opentrons OT-2 digital twin.

This environment exposes the OT-2 pipette motion task as a reinforcement learning
problem. An agent controls the pipette tip via normalized Cartesian velocity
commands and must move it to a randomly sampled goal position within the measured
work envelope.

The wrapper defines the action space, observation space, reward function, and
termination conditions required to train RL agents using Stable Baselines 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p

from sim_class import Simulation


@dataclass(frozen=True)
class WorkEnvelope:
    """
    Measured OT-2 pipette work envelope (axis-aligned box) from Task 9.

    Units: meters (m)

    These bounds are used for:
    - Sampling random goal positions inside the reachable workspace
    - Normalizing positions to a stable [-1, 1] observation range for RL
    """
    # Work Envelope bounds
    x_min: float = -0.1870
    x_max: float =  0.2530
    y_min: float = -0.1705
    y_max: float =  0.2195
    z_min: float =  0.1695
    z_max: float =  0.2895


    def sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        """Uniformly sample a goal position inside the envelope."""
        gx = rng.uniform(self.x_min, self.x_max)
        gy = rng.uniform(self.y_min, self.y_max)
        gz = rng.uniform(self.z_min, self.z_max)
        return np.array([gx, gy, gz], dtype=np.float32)
    
    def sample_goal_box(self, rng: np.random.Generator, center: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
        """Sample uniformly in an axis-aligned box centered at `center` with half side lengths `half_extent`."""
        lo = np.maximum(center - half_extent, np.array([self.x_min, self.y_min, self.z_min], dtype=np.float32))
        hi = np.minimum(center + half_extent, np.array([self.x_max, self.y_max, self.z_max], dtype=np.float32))
        g = rng.uniform(lo, hi)
        return g.astype(np.float32)

    @property
    def center(self) -> np.ndarray:
        """Center of the box."""
        return np.array(
            [
                (self.x_min + self.x_max) / 2.0,
                (self.y_min + self.y_max) / 2.0,
                (self.z_min + self.z_max) / 2.0,
            ],
            dtype=np.float32,
        )

    @property
    def half_range(self) -> np.ndarray:
        """
        Half the side length per axis.

        Used for normalization:
            normalized = (pos - center) / half_range
        which maps the envelope bounds approximately to [-1, 1].
        """
        return np.array(
            [
                (self.x_max - self.x_min) / 2.0,
                (self.y_max - self.y_min) / 2.0,
                (self.z_max - self.z_min) / 2.0,
            ],
            dtype=np.float32,
        )

    @property
    def diagonal_length(self) -> float:
        """Length of the workspace diagonal (useful to scale rewards)."""
        return float(np.linalg.norm(self.half_range * 2.0))

    def normalize(self, pos: np.ndarray) -> np.ndarray:
        """
        Normalize a position to approximately [-1, 1]^3 based on envelope bounds.
        """
        hr = np.maximum(self.half_range, 1e-9)  # avoid division by zero
        return ((pos - self.center) / hr).astype(np.float32)



class OT2GymEnv(gym.Env):
    """
    Gymnasium environment wrapper for the OT-2 digital twin (PyBullet) via `Simulation`.

    Task definition:
        Move the OT-2 pipette tip to a randomly sampled goal position within the work envelope.

    Action space:
        a = [ax, ay, az] in [-1, 1]^3 (normalized)
        These are scaled to real velocities (m/s) internally.

    Observation space (9D):
        [pip_norm(3), goal_norm(3), delta_norm(3)]
        where delta_norm = goal_norm - pip_norm

    Termination:
        - terminated = True if distance_to_goal < success_threshold_m
        - truncated  = True if steps >= max_steps
        - optional early stuck termination (disabled by default; can cause "corner camping")

    Reward (dense, stable):
        reward = progress_term + distance_weight * distance_term - step_penalty
        plus success_bonus on success.

    Debug:
        If render=True, a small green sphere marks the goal position.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        # Episode config
        max_steps: int = 400,                               # Max steps per episode
        success_threshold_m: float = 0.005,                 # Success threshold in meters
        # Action scaling
        vel_xy_max: float = 0.25,                            # Max XY velocity (m/s)     
        vel_z_max: float = 0.15,                             # Max Z velocity (m/s)
        # RNG / rendering
        seed: Optional[int] = None,                         # RNG seed
        render: bool = False,                               # Render PyBullet GUI   
        # Reward shaping
        step_penalty: float = 0.001,                        # Small per-step penalty to encourage faster completion
        success_bonus: float = 10.0,                        # Large bonus for reaching the goal
        progress_scale: float = 20.0,                       # Scale for progress term in reward
        distance_weight: float = 1.0,                       # Weight for distance term in reward
        action_penalty_coeff: float = 0.0005,               # Coefficient for action smoothness penalty
        # Stuck termination (training convenience)
        enable_stuck_termination: bool = False,             # Whether to enable early termination for being "stuck" / # IMPORTANT: disable to avoid "stall/corner" local optimum
        stuck_min_steps: int = 150,                         # Minimum steps before stuck termination can occur
        stuck_patience_steps: int = 120,                    # Steps without improvement to consider "stuck"
        stuck_min_improvement_m: float = 0.0005,            # Minimum improvement to avoid being considered "stuck"
        stuck_penalty: float = 0.15,                        # Penalty applied on stuck termination
        # Goal marker (render/debug)
        show_goal_marker: bool = True,                      # Whether to show a goal marker when rendering
        goal_marker_radius: float = 0.004,                  # Radius of the goal marker sphere
        # Goal curriculum (optional)
        goal_curriculum_half_extent: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()

        # Work envelope
        self.envelope = WorkEnvelope()

        # Episode config
        self.max_steps = int(max_steps)
        self.success_threshold_m = float(success_threshold_m)

        # Action scaling
        self.vel_xy_max = float(vel_xy_max)
        self.vel_z_max = float(vel_z_max)

        # RNG / rendering
        self._rng = np.random.default_rng(seed)

        # Simulation instance (single agent)
        self._render = bool(render)
        self.sim = Simulation(num_agents=1, render=self._render)

        # Action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space (9D): pip_norm, goal_norm, delta_norm
        # pip_norm and goal_norm are approximately in [-1,1]. delta can be roughly [-2,2].
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(9,), dtype=np.float32)

        # Reward shaping config
        self.step_penalty = float(step_penalty)
        self.success_bonus = float(success_bonus)
        self.progress_scale = float(progress_scale)
        self.distance_weight = float(distance_weight)
        self.action_penalty_coeff = float(action_penalty_coeff)
        self._diag = max(self.envelope.diagonal_length, 1e-9)

        # Stuck termination config
        self.enable_stuck_termination = bool(enable_stuck_termination)
        self.stuck_min_steps = int(stuck_min_steps)
        self.stuck_patience_steps = int(stuck_patience_steps)
        self.stuck_min_improvement_m = float(stuck_min_improvement_m)
        self.stuck_penalty = float(stuck_penalty)

        # Per-episode state
        self.steps: int = 0
        self.goal_position: np.ndarray = np.zeros(3, dtype=np.float32)
        self.prev_dist: Optional[float] = None

        # Stuck tracking state
        self._best_dist: float = float("inf")
        self._steps_since_best: int = 0

        # Goal marker state
        self._show_goal_marker = bool(show_goal_marker)
        self._goal_marker_radius = float(goal_marker_radius)
        self._goal_marker_body_id: Optional[int] = None

        # Goal curriculum
        self.goal_curriculum_half_extent = goal_curriculum_half_extent


    # -------------------------
    # Helpers
    # -------------------------
    def _get_pipette_position(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract pipette tip (x,y,z) from Simulation state dict.

        The simulation returns a nested dict keyed by something like 'robotId_1'.
        """
        robot_key = list(state.keys())[0]
        pos = state[robot_key]["pipette_position"]
        return np.array([pos[0], pos[1], pos[2]], dtype=np.float32)


    def _scale_action_to_velocity(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action in [-1,1] to real velocities [vx,vy,vz]."""
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        vx = a[0] * self.vel_xy_max
        vy = a[1] * self.vel_xy_max
        vz = a[2] * self.vel_z_max
        return np.array([vx, vy, vz], dtype=np.float32)
    

    def _make_obs(self, pipette_pos: np.ndarray) -> np.ndarray:
        """
        Build normalized observation (9D):
            [pip_norm(3), goal_norm(3), delta_norm(3)]
        """
        pip_n = self.envelope.normalize(pipette_pos)
        goal_n = self.envelope.normalize(self.goal_position)
        delta_n = goal_n - pip_n
        return np.concatenate([pip_n, goal_n, delta_n]).astype(np.float32)
    

    def _update_goal_marker(self) -> None:
        """Create or move a small marker to show the goal position when render=True."""
        # No marker if not rendering or disabled
        if not self._render or not self._show_goal_marker or p is None:
            return

        # Create or move marker sphere
        try:
            # Create marker if it doesn't exist
            if self._goal_marker_body_id is None:
                visual = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=self._goal_marker_radius,
                    rgbaColor=[0.0, 1.0, 0.0, 0.8],  # green, semi-transparent
                )
                collision = -1  # visual only
                self._goal_marker_body_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=self.goal_position.tolist(),
                )
            # Move existing marker
            else:
                p.resetBasePositionAndOrientation(
                    self._goal_marker_body_id,
                    self.goal_position.tolist(),
                    [0.0, 0.0, 0.0, 1.0],
                )
        except Exception:
            # If anything goes wrong, just disable marker silently
            self._goal_marker_body_id = None    


    def _is_inside_envelope(self, pos: np.ndarray) -> bool:
        """Return True if pos is inside the measured work envelope bounds."""
        return bool(
            (self.envelope.x_min <= pos[0] <= self.envelope.x_max) and
            (self.envelope.y_min <= pos[1] <= self.envelope.y_max) and
            (self.envelope.z_min <= pos[2] <= self.envelope.z_max)
        )
    

    def _force_start_inside_envelope(self) -> np.ndarray:
        """
        Force the simulator start pose to a known, valid pipette position.
        Uses Simulation.set_start_position(...) and returns the resulting pipette position.
        """
        # Use envelope center as safe start position
        start = self.envelope.center.copy()

        # Set start pose (x,y,z) to envelope center
        self.sim.set_start_position(float(start[0]), float(start[1]), float(start[2]))

        # One no-op step to let the sim report updated state
        state = self.sim.run([[0.0, 0.0, 0.0, 0]])

        # Return the actual pipette position after setting start
        return self._get_pipette_position(state)
    

    # -------------------------
    # Gymnasium API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns:
            obs: Initial observation
            info: Additional info dict
        """
        # Set RNG seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.prev_dist = None

        # Sample new goal (optionally curriculum-restricted)
        if self.goal_curriculum_half_extent is None:
            self.goal_position = self.envelope.sample_goal(self._rng)
        else:
            he = np.array(self.goal_curriculum_half_extent, dtype=np.float32)
            self.goal_position = self.envelope.sample_goal_box(self._rng, self.envelope.center, he)


        # Reset sim
        self.sim.reset(num_agents=1)

        # Force start pose inside the work envelope (fixes Z-min reset bug) and get actual pipette pos
        pipette_pos = self._force_start_inside_envelope()

        # Ensure start pose is inside envelope
        if not self._is_inside_envelope(pipette_pos):
            raise RuntimeError(f"Reset pipette pos outside envelope: {pipette_pos}")

        # Ensure marker is visible at the correct spot (render=True only)
        self._update_goal_marker()
        
        # Compute initial distance to goal
        dist = float(np.linalg.norm(pipette_pos - self.goal_position))
        self.prev_dist = dist

        # Initialize stuck tracking state
        self._best_dist = dist
        # Reset stuck tracking state
        self._steps_since_best = 0

        # Build initial observation
        obs = self._make_obs(pipette_pos)

        # Build info dict
        info: Dict[str, Any] = {
            "goal_position": self.goal_position.copy(),     # Goal position
            "pipette_position": pipette_pos.copy(),         # Current pipette position
            "distance_to_goal": dist,                       # Current distance to goal
        }
        return obs, info


    def step(self, action: np.ndarray):
        self.steps += 1

        v = self._scale_action_to_velocity(action)
        drop_command = 0
        sim_action = [float(v[0]), float(v[1]), float(v[2]), drop_command]
        state = self.sim.run([sim_action])

        pipette_pos = self._get_pipette_position(state)
        dist = float(np.linalg.norm(pipette_pos - self.goal_position))

        # --- progress ---
        progress = 0.0 if self.prev_dist is None else (self.prev_dist - dist)
        self.prev_dist = dist

        # --- base dense reward: normalize by workspace size for stability ---
        # Normalize distance and progress by workspace diagonal
        dist_norm = dist / self._diag
        # Negative distance term (closer = less negative)
        r_dist = -dist_norm

        # Progress shaping: also normalize (progress is meters per step)
        progress_norm = progress / self._diag
        r_progress = self.progress_scale * progress_norm

        # Step cost: encourages faster solutions
        r_step = -self.step_penalty

        # Action smoothness penalty
        r_action = -self.action_penalty_coeff * float(np.sum(np.square(action)))

        # Workspace bound penalty (soft): if outside envelope, push back
        # This should almost never happen if envelope is correct, but helps catch drift.
        outside = 0.0
        if (
            # Check if pipette position is outside the workspace bounds
            pipette_pos[0] < self.envelope.x_min or pipette_pos[0] > self.envelope.x_max or
            pipette_pos[1] < self.envelope.y_min or pipette_pos[1] > self.envelope.y_max or
            pipette_pos[2] < self.envelope.z_min or pipette_pos[2] > self.envelope.z_max
        ):
            # Outside workspace bounds
            outside = 1.0
        # Negative reward proportional to how far outside (0.05 m scale)
        r_outside = -0.05 * outside

        # --- total reward ---
        reward = (self.distance_weight * r_dist) + r_progress + r_step + r_action + r_outside

        # --- termination / truncation ---
        terminated_success = dist < self.success_threshold_m
        terminated = bool(terminated_success)
        if terminated_success:
            reward += self.success_bonus

        truncated = self.steps >= self.max_steps

        # --- best dist tracking ---
        prev_best = self._best_dist
        if dist < self._best_dist:
            self._best_dist = dist

        # --- stuck termination (optional) ---
        terminated_stuck = False
        if self.enable_stuck_termination and (not terminated) and (not truncated):
            if dist < prev_best - self.stuck_min_improvement_m:
                self._steps_since_best = 0
            else:
                self._steps_since_best += 1

            if (self.steps >= self.stuck_min_steps) and (self._steps_since_best >= self.stuck_patience_steps):
                terminated_stuck = True
                terminated = True
                reward -= self.stuck_penalty

        obs = self._make_obs(pipette_pos)

        info: Dict[str, Any] = {
            "goal_position": self.goal_position.copy(),         # Goal position
            "pipette_position": pipette_pos.copy(),             # Current pipette position
            "distance_to_goal": dist,                           # Distance to goal
            "steps": self.steps,                                # Number of steps taken
            "terminated_success": bool(terminated_success),     # Whether terminated successfully
            "terminated_stuck": bool(terminated_stuck),         # Whether terminated due to being stuck
            "drop_command": drop_command,                       # Drop command status  
            "best_distance_to_goal": float(self._best_dist),    # Best distance achieved so far

            # Detailed reward components for debugging
            "reward_components": {
                "r_dist": float(r_dist),                # negative distance term
                "r_progress": float(r_progress),        # progress shaping term
                "r_step": float(r_step),                # step penalty
                "r_action": float(r_action),            # action smoothness penalty
                "r_outside": float(r_outside),          # outside workspace penalty
                "progress_m": float(progress),          # raw progress in meters
                "progress_norm": float(progress_norm),  # normalized progress
                "dist_m": float(dist),                  # raw distance in meters
                "dist_norm": float(dist_norm),          # normalized distance
                "outside": float(outside),              # outside workspace indicator
            }
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    
    def close(self) -> None:
        """Clean up resources when closing the environment."""
        # Remove marker if it exists
        if p is not None and self._goal_marker_body_id is not None:
            try:
                p.removeBody(self._goal_marker_body_id)
            except Exception:
                pass
            self._goal_marker_body_id = None

        # Close simulation connection
        try:
            self.sim.close()
        except Exception:
            pass

        super().close()    