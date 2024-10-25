import numpy
import mazes


class EnvManager:
    def __init__(
        self, size: list[int], num_games: int, use_maze: bool, use_static_goals
    ) -> None:
        self.size = size
        self.use_static_goals = use_static_goals
        self.use_maze = use_maze
        self.num_games = num_games
        self.env = mazes.Maze(
            self.size, self.num_games, self.use_maze, self.use_static_goals
        )
        self.update_map()

    def reset_env(self):
        self.env.restart(self.num_games)
        self.update_map()

    def make_move(self, actions):
        self.reset_maze()
        rewards, ends = self.env.move(actions)
        self.update_map()
        return rewards, ends

    def reset_maze(self):
        num_games, num_positions, _ = self.env.game_positions.shape

        game_indices = numpy.repeat(numpy.arange(num_games), num_positions)
        x_indices = self.env.game_positions[:, :, 0].ravel()
        y_indices = self.env.game_positions[:, :, 1].ravel()

        self.env.mazes[game_indices, x_indices, y_indices] = 0

    def update_map(self):
        num_games, _, _ = self.env.game_positions.shape
        game_indices = numpy.arange(num_games)

        sub_goal_x_indices = self.env.game_positions[:, 1, 0].ravel()
        sub_goal_y_indices = self.env.game_positions[:, 1, 1].ravel()
        goal_x_indices = self.env.game_positions[:, 2, 0].ravel()
        goal_y_indices = self.env.game_positions[:, 2, 1].ravel()

        player_pos_x_indices = self.env.game_positions[:, 0, 0]
        player_pos_y_indices = self.env.game_positions[:, 0, 1]

        self.env.mazes[
            game_indices,
            goal_x_indices,
            goal_y_indices,
        ] = 4

        self.env.mazes[
            game_indices[~self.env.sub_goal_reached_flags],
            sub_goal_x_indices[~self.env.sub_goal_reached_flags],
            sub_goal_y_indices[~self.env.sub_goal_reached_flags],
        ] = 3

        self.env.mazes[game_indices, player_pos_x_indices, player_pos_y_indices] = 2

    def remove_ended_games(self, ended_game_idx):
        self.env.mazes = numpy.delete(self.env.mazes, ended_game_idx, axis=0)
        self.env.sub_goal_reached_flags = numpy.delete(
            self.env.sub_goal_reached_flags, ended_game_idx, axis=0
        )
        self.env.game_positions = numpy.delete(
            self.env.game_positions, ended_game_idx, axis=0
        )
