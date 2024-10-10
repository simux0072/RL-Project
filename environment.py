import numpy
from maze import Maze


class EnvManager:
    def __init__(self) -> None:
        self.env = Maze()
        self.maze = numpy.ones((10, 10))
        self.maze[1:9, 1:9] = 0
        self.update_map()

    def update_map(self):
        for idx in range(len(self.env.goals)):
            self.maze[self.env.goals[idx, 0], self.env.goals[idx, 1]] = idx + 2
        self.maze[self.env.player_pos[0], self.env.player_pos[1]] = 4

    def reset_env(self):
        self.maze[self.env.player_pos[0], self.env.player_pos[1]] = 0
        self.maze[self.env.goals[:, 0], self.env.goals[:, 1]] = 0
        self.env.restart()
        self.update_map()

    def get_state(self):
        return self.maze

    def make_move(self, action: int):
        self.maze[self.env.player_pos[0], self.env.player_pos[1]] = 0
        reward, end = self.env.move(action)
        self.maze[self.env.goals[1, 0], self.env.goals[1, 1]] = 3
        self.maze[self.env.player_pos[0], self.env.player_pos[1]] = 4
        return reward, end
