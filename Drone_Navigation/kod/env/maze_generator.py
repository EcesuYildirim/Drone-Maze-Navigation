import numpy as np
import pybullet as p


MAZE_LAYOUTS = {
    "easy": [
        "###########",
        "#S  #     #",
        "# # # ### #",
        "# #   #   #",
        "# ##### # #",
        "#       # #",
        "# ####### #",
        "#         #",
        "##### ### #",
        "#       #G#",
        "###########",
    ],
    "medium": [
        "#############",
        "#S  #   #   #",
        "# # # # # # #",
        "# #   #   # #",
        "# ##### ### #",
        "#     #     #",
        "### # # ### #",
        "#   #   #   #",
        "# ####### # #",
        "#       # # #",
        "####### # # #",
        "#         #G#",
        "#############",
    ],
    "hard": [
        "###############",
        "#S  #   # #   #",
        "# # ### # # # #",
        "# #   #   # # #",
        "# ### # ### # #",
        "#     #   #   #",
        "# ####### # ###",
        "# #   #   #   #",
        "# # # # ##### #",
        "#   #   #   # #",
        "### ####### # #",
        "#   #       # #",
        "# ### ####### #",
        "#           #G#",
        "###############",
    ],
}


class MazeGenerator:
    """
    PyBullet'te grid tabanlı 3D labirent oluşturur.
    '#' → duvar kutusu, 'S' → başlangıç, 'G' → hedef, ' ' → boş
    """

    def __init__(self, physics_client: int, cell_size: float = 1.0, wall_height: float = 1.5):
        self.client = physics_client
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.wall_ids: list[int] = []
        self.start_pos: np.ndarray = np.zeros(3)
        self.goal_pos: np.ndarray = np.zeros(3)

    def load(self, layout_name: str = "medium", random_goal: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Labirenti yükler. start_pos ve goal_pos döndürür.
        random_goal=True ise boş hücrelerden rastgele hedef seçilir.
        """
        self._clear()
        self.empty_positions: list[np.ndarray] = []

        grid = MAZE_LAYOUTS[layout_name]
        rows = len(grid)

        empty_cells: list[tuple[int, int]] = []

        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                x = c * self.cell_size
                y = (rows - 1 - r) * self.cell_size  # y yukarı artar
                z = self.wall_height / 2.0

                if ch == "#":
                    self._create_wall(x, y, z)
                elif ch == "S":
                    self.start_pos = np.array([x, y, 0.5])
                    self.empty_positions.append(np.array([x, y, 0.5]))
                elif ch == "G":
                    self.goal_pos = np.array([x, y, 0.5])
                elif ch == " ":
                    empty_cells.append((r, c))
                    self.empty_positions.append(np.array([x, y, 0.5]))

        if random_goal and empty_cells:
            r, c = empty_cells[np.random.randint(len(empty_cells))]
            x = c * self.cell_size
            y = (rows - 1 - r) * self.cell_size
            self.goal_pos = np.array([x, y, 0.5])

        return self.start_pos.copy(), self.goal_pos.copy()

    def get_empty_positions(self) -> list[np.ndarray]:
        """Boş ve başlangıç hücrelerinin 3D pozisyonlarını döndürür."""
        return list(self.empty_positions)

    def _create_wall(self, x: float, y: float, z: float) -> None:
        half = self.cell_size / 2.0
        half_h = self.wall_height / 2.0

        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half, half, half_h],
            physicsClientId=self.client,
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half, half, half_h],
            rgbaColor=[0.4, 0.4, 0.4, 1.0],
            physicsClientId=self.client,
        )
        wall_id = p.createMultiBody(
            baseMass=0,                         # statik cisim
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[x, y, z],
            physicsClientId=self.client,
        )
        self.wall_ids.append(wall_id)

    def create_goal_marker(self) -> int:
        """Hedef konumuna görsel bir küre koyar (çarpışma yok)."""
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.2,
            rgbaColor=[0.0, 1.0, 0.0, 0.6],
            physicsClientId=self.client,
        )
        marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_shape,
            basePosition=self.goal_pos.tolist(),
            physicsClientId=self.client,
        )
        return marker_id

    def get_wall_ids(self) -> list[int]:
        return list(self.wall_ids)

    def _clear(self) -> None:
        for wid in self.wall_ids:
            p.removeBody(wid, physicsClientId=self.client)
        self.wall_ids.clear()
