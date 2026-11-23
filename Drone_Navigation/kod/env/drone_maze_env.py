"""
DroneMazeEnv — Güncellenmiş versiyon
Yeni özellikler:
  - altitude_penalty_scale: z=0.5'ten saptığında ceza
  - hover_bonus_scale: z=0.5 civarında (±0.2m) kalınca pozitif ödül

Orijinal kodla %100 geriye uyumlu: her iki parametre 0.0 → eski davranış.
"""

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .maze_generator import MazeGenerator, MAZE_LAYOUTS


class DroneMazeEnv(BaseAviary):
    """
    PPO ile eğitilecek İHA labirent navigasyon ortamı.

    Observation : 39-boyutlu sürekli vektör
        [0:3]   konum        (x, y, z)
        [3:6]   çizgisel hız (vx, vy, vz)
        [6:9]   açısal hız   (wx, wy, wz)
        [9:12]  yönelim      (roll, pitch, yaw)
        [12:36] LIDAR        (24 ışın mesafe değeri)
        [36:39] hedef vektörü (goal_x - x, goal_y - y, goal_z - z)

    Action : 4-boyutlu sürekli [0.0, 1.0]
        Her bileşen bir pervane motor itkisini temsil eder.

    Reward :
        +k*(d_prev - d_now)            yaklaşma ödülü
        +100                           hedefe ulaşma bonusu
        -100                           çarpışma cezası
        -0.01                          adım enerji cezası
        -altitude_penalty_scale*|z-0.5|  yükseklik sapma cezası (YENİ)
    """

    NUM_LIDAR_RAYS  = 24
    LIDAR_MAX_RANGE = 5.0
    GOAL_RADIUS     = 0.3
    MAX_STEPS       = 1000

    REWARD_GOAL      =  100.0
    REWARD_COLLISION = -100.0
    REWARD_STEP      =   -0.01
    APPROACH_SCALE         =    1.0
    ALTITUDE_PENALTY_SCALE =    0.0
    HOVER_BONUS_SCALE      =    0.0
    DIST_PENALTY_SCALE     =    0.0   # her adımda -scale*curr_dist → uzakta durmayı pahalı yapar
    TARGET_ALTITUDE        =    0.5

    def __init__(
        self,
        layout: str = "medium",
        random_goal: bool = False,
        gui: bool = False,
        record: bool = False,
        # --- Deney parametreleri (opsiyonel) ---
        approach_scale: float = None,
        altitude_penalty_scale: float = None,
        hover_bonus_scale: float = None,
        dist_penalty_scale: float = None,
        collision_penalty: float = None,
        goal_reward: float = None,
        step_penalty: float = None,
        max_steps: int = None,
    ):
        self.layout      = layout
        self.random_goal = random_goal

        # Deney parametrelerini uygula (verilmezse sınıf varsayılanı kalır)
        if approach_scale is not None:
            self.APPROACH_SCALE = approach_scale
        if altitude_penalty_scale is not None:
            self.ALTITUDE_PENALTY_SCALE = altitude_penalty_scale
        if hover_bonus_scale is not None:
            self.HOVER_BONUS_SCALE = hover_bonus_scale
        if dist_penalty_scale is not None:
            self.DIST_PENALTY_SCALE = dist_penalty_scale
        if collision_penalty is not None:
            self.REWARD_COLLISION = collision_penalty
        if goal_reward is not None:
            self.REWARD_GOAL = goal_reward
        if step_penalty is not None:
            self.REWARD_STEP = step_penalty
        if max_steps is not None:
            self.MAX_STEPS = max_steps

        # İç durum
        self._goal_pos       = np.zeros(3)
        self._prev_dist      = 0.0
        self._step_count     = 0
        self._wall_ids       = []
        self._last_collision = False   # _computeReward'da hesaplanır, _computeTerminated yeniden kullanır
        # Curriculum: None → orijinal başlangıç, float → hedef etrafında spawn yarıçapı
        self._spawn_radius: float | None = None
        self._empty_cells: list[np.ndarray] = []

        start_xyz = self._find_start_pos(layout)

        super().__init__(
            drone_model    = DroneModel.CF2X,
            num_drones     = 1,
            physics        = Physics.PYB,
            initial_xyzs   = np.array([start_xyz]),
            gui            = gui,
            record         = record,
            obstacles      = True,
        )

    # ------------------------------------------------------------------ #
    # BaseAviary zorunlu: action & observation space                       #
    # ------------------------------------------------------------------ #
    def _actionSpace(self):
        return spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        low  = np.full(39, -np.inf, dtype=np.float32)
        high = np.full(39,  np.inf, dtype=np.float32)
        low[12:36]  = 0.0
        high[12:36] = self.LIDAR_MAX_RANGE
        return spaces.Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Engel yükleme                                                       #
    # ------------------------------------------------------------------ #
    def _addObstacles(self):
        maze = MazeGenerator(
            physics_client = self.CLIENT,
            cell_size      = 1.0,
            wall_height    = 1.5,
        )
        start_pos, goal_pos = maze.load(self.layout, random_goal=self.random_goal)
        self._wall_ids    = maze.get_wall_ids()
        self._goal_pos    = goal_pos
        self._empty_cells = maze.get_empty_positions()
        maze.create_goal_marker()

        drone_start = np.array([start_pos[0], start_pos[1], self.TARGET_ALTITUDE])
        self._prev_dist  = float(np.linalg.norm(drone_start - self._goal_pos))
        self._step_count = 0

    # ------------------------------------------------------------------ #
    # Gözlem (39-boyut)                                                    #
    # ------------------------------------------------------------------ #
    def _computeObs(self) -> np.ndarray:
        pos      = self.pos[0]
        vel      = self.vel[0]
        angv     = self.ang_v[0]
        rpy      = self.rpy[0]
        lidar    = self._compute_lidar()
        goal_vec = (self._goal_pos - pos).astype(np.float32)
        return np.concatenate([pos, vel, angv, rpy, lidar, goal_vec]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # LIDAR (24 ışın)                                                      #
    # ------------------------------------------------------------------ #
    def _compute_lidar(self) -> np.ndarray:
        drone_pos = self.pos[0]
        angles    = np.linspace(0, 2 * np.pi, self.NUM_LIDAR_RAYS, endpoint=False)

        ray_froms, ray_tos = [], []
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            ray_froms.append(drone_pos.tolist())
            ray_tos.append((drone_pos + direction * self.LIDAR_MAX_RANGE).tolist())

        results   = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.CLIENT)
        distances = np.array(
            [hit[2] * self.LIDAR_MAX_RANGE if hit[0] != -1 else self.LIDAR_MAX_RANGE
             for hit in results],
            dtype=np.float32,
        )
        return distances

    # ------------------------------------------------------------------ #
    # Ödül (güncellenmiş — yükseklik cezası eklendi)                      #
    # ------------------------------------------------------------------ #
    def _computeReward(self) -> float:
        curr_dist       = float(np.linalg.norm(self.pos[0] - self._goal_pos))
        approach        = self.APPROACH_SCALE * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist

        reward = approach + self.REWARD_STEP

        # Mesafe cezası: hedefe uzakta durmayı pahalı yapar, hover lokal optimumunu kırar
        if self.DIST_PENALTY_SCALE > 0:
            reward -= self.DIST_PENALTY_SCALE * curr_dist

        alt_error = abs(self.pos[0][2] - self.TARGET_ALTITUDE)

        # Yükseklik sapma cezası
        if self.ALTITUDE_PENALTY_SCALE > 0:
            reward -= self.ALTITUDE_PENALTY_SCALE * alt_error

        # Hover bonusu: z=0.5'e yakın kalınca pozitif ödül (max 0.2m tolerans)
        if self.HOVER_BONUS_SCALE > 0:
            hover_tol = 0.2
            if alt_error < hover_tol:
                reward += self.HOVER_BONUS_SCALE * (hover_tol - alt_error) / hover_tol

        self._last_collision = self._check_collision()

        if curr_dist < self.GOAL_RADIUS:
            reward += self.REWARD_GOAL
        elif self._last_collision:
            reward += self.REWARD_COLLISION

        return float(reward)

    # ------------------------------------------------------------------ #
    # Sonlanma koşulları                                                   #
    # ------------------------------------------------------------------ #
    def _computeTerminated(self) -> bool:
        at_goal = np.linalg.norm(self.pos[0] - self._goal_pos) < self.GOAL_RADIUS
        return bool(at_goal or self._last_collision)

    def _computeTruncated(self) -> bool:
        return self._step_count >= self.MAX_STEPS

    def _computeInfo(self) -> dict:
        return {
            "distance_to_goal": float(np.linalg.norm(self.pos[0] - self._goal_pos)),
            "step": self._step_count,
        }

    # ------------------------------------------------------------------ #
    # Curriculum desteği                                                  #
    # ------------------------------------------------------------------ #
    def set_curriculum(self, layout: str | None = None, spawn_radius: float | None = None) -> None:
        """Callback tarafından çağrılır; bir sonraki reset()'ten itibaren geçerli olur."""
        if layout is not None:
            self.layout = layout
        self._spawn_radius = spawn_radius

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if self._spawn_radius is not None:
            spawn = self._sample_spawn_near_goal()
            if spawn is not None:
                self._teleport_to(spawn)
                obs = self._computeObs()
        return obs, info

    def _teleport_to(self, pos: np.ndarray) -> None:
        drone_id = self.DRONE_IDS[0]
        orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=self.CLIENT)[1]
        p.resetBasePositionAndOrientation(drone_id, pos.tolist(), orn, physicsClientId=self.CLIENT)
        p.resetBaseVelocity(drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.CLIENT)
        self._updateAndStoreKinematicInformation()
        self._prev_dist = float(np.linalg.norm(pos - self._goal_pos))

    def _sample_spawn_near_goal(self) -> np.ndarray | None:
        """_spawn_radius içinde, hedeften en az GOAL_RADIUS uzakta rastgele boş hücre seçer."""
        goal_xy = self._goal_pos[:2]
        r = self._spawn_radius
        candidates = [
            pos for pos in self._empty_cells
            if self.GOAL_RADIUS < np.linalg.norm(pos[:2] - goal_xy) <= r
        ]
        if not candidates:
            return None
        return candidates[np.random.randint(len(candidates))]

    # ------------------------------------------------------------------ #
    # Step                                                                 #
    # ------------------------------------------------------------------ #
    def step(self, action):
        self._step_count += 1
        return super().step(action)

    # ------------------------------------------------------------------ #
    # Çarpışma kontrolü                                                    #
    # ------------------------------------------------------------------ #
    def _check_collision(self) -> bool:
        for wall_id in self._wall_ids:
            if p.getContactPoints(
                bodyA=self.DRONE_IDS[0],
                bodyB=wall_id,
                physicsClientId=self.CLIENT,
            ):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Aksiyon ön işleme: [0,1] → RPM                                      #
    # ------------------------------------------------------------------ #
    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        # Shift so action=0.5 → hover RPM; untrained policy can hover from the start
        hover_frac = self.HOVER_RPM / self.MAX_RPM   # ~0.667 for CF2X
        rpm = np.clip(action + hover_frac - 0.5, 0.0, 1.0) * self.MAX_RPM
        return rpm.reshape(1, 4)

    # ------------------------------------------------------------------ #
    # Yardımcı                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_start_pos(layout: str) -> list:
        grid = MAZE_LAYOUTS[layout]
        rows = len(grid)
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    return [float(c), float(rows - 1 - r), 0.5]
        return [1.0, 1.0, 0.5]
