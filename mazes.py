import numpy

ACTIONS = numpy.array(
    [[1, 0], [-1, 0], [0, 1], [0, -1]]
)  # 0 - down; 1 - up; 2 - right; 3- left

POSSIBLE_LOCATIONS = numpy.array(
    [
        [1, 1],
        [2, 1],
        [3, 1],
        [5, 1],
        [6, 1],
        [8, 1],
        [3, 2],
        [6, 2],
        [8, 2],
        [1, 3],
        [2, 3],
        [3, 3],
        [4, 3],
        [5, 3],
        [6, 3],
        [7, 3],
        [8, 3],
        [1, 4],
        [5, 4],
        [8, 4],
        [1, 5],
        [2, 5],
        [3, 5],
        [5, 5],
        [7, 5],
        [8, 5],
        [3, 6],
        [5, 6],
        [6, 6],
        [1, 7],
        [2, 7],
        [3, 7],
        [6, 7],
        [8, 7],
        [1, 8],
        [3, 8],
        [4, 8],
        [5, 8],
        [6, 8],
        [7, 8],
        [8, 8],
    ]
)

STATIC_GOALS = numpy.array([[1, 1], [5, 3], [8, 7]])


class Maze:
    def __init__(
        self, size: list[int], num_games: int, use_maze: bool, use_static_goals: bool
    ) -> None:
        self.use_maze = use_maze
        self.use_static_goals = use_static_goals
        self.possible_locations = POSSIBLE_LOCATIONS.copy()
        self.goal_mesh = numpy.array(
            numpy.meshgrid([range(1, size[0] - 1)], [range(1, size[1] - 1)])
        ).T.reshape(-1, 2)
        self.size = size
        self.num_games = num_games
        self.restart(self.num_games)

    def restart(self, num_games: int):
        self.mazes = numpy.ones((num_games, self.size[0], self.size[1]))
        if self.use_maze:
            self.mazes[
                :, self.possible_locations[:, 0], self.possible_locations[:, 1]
            ] = 0
        else:
            self.mazes[
                :,
                1 : self.mazes.shape[1] - 1,
                1 : self.mazes.shape[2] - 1,
            ] = 0
        self.game_positions = numpy.zeros((num_games, 3, 2), dtype=int)
        self.sub_goal_reached_flags = numpy.zeros(num_games, dtype=bool)
        self.generate_goals(num_games)

    def generate_goals(self, num_games: int):
        for idx in range(num_games):
            if self.use_maze:
                if self.use_static_goals:
                    self.game_positions[idx] = STATIC_GOALS.copy()
                else:
                    numpy.random.shuffle(self.possible_locations)
                    self.game_positions[idx] = self.possible_locations[:3].copy()
            else:
                numpy.random.shuffle(self.goal_mesh)
                self.game_positions[idx] = self.goal_mesh[:3].copy()

    def move(self, actions):
        index_list = numpy.arange(self.game_positions.shape[0])
        results = numpy.zeros(self.game_positions.shape[0])

        new_positions = self.game_positions[:, 0] + ACTIONS[actions]
        has_not_hit_wall = (
            self.mazes[index_list, new_positions[:, 0], new_positions[:, 1]] != 1
        )
        self.game_positions[has_not_hit_wall, 0] = new_positions[has_not_hit_wall]
        results[has_not_hit_wall == False] = -1

        return self.check_goal(results)

    def create_mask(self):
        index_list = numpy.arange(self.game_positions.shape[0])

        down_new_pos = self.game_positions[:, 0] + ACTIONS[0]
        up_new_pos = self.game_positions[:, 0] + ACTIONS[1]
        right_new_pos = self.game_positions[:, 0] + ACTIONS[2]
        left_new_pos = self.game_positions[:, 0] + ACTIONS[3]

        down_not_wall = (
            self.mazes[index_list, down_new_pos[:, 0], down_new_pos[:, 1]] != 1
        )
        up_not_wall = self.mazes[index_list, up_new_pos[:, 0], up_new_pos[:, 1]] != 1
        right_not_wall = (
            self.mazes[index_list, right_new_pos[:, 0], right_new_pos[:, 1]] != 1
        )
        left_not_wall = (
            self.mazes[index_list, left_new_pos[:, 0], left_new_pos[:, 1]] != 1
        )
        masks = numpy.stack(
            [down_not_wall, up_not_wall, right_not_wall, left_not_wall], axis=1
        )
        return masks

    def check_goal(self, results):
        ends = numpy.zeros(self.game_positions.shape[0], dtype=bool)
        sub_goal_reached = numpy.all(
            self.game_positions[:, 0] == self.game_positions[:, 1], axis=1
        )
        goal_reached = numpy.all(
            self.game_positions[:, 0] == self.game_positions[:, 2], axis=1
        )
        results[sub_goal_reached & (self.sub_goal_reached_flags == 0)] = 10
        # results[self.sub_goal_reached_flags == 1] = 0
        self.sub_goal_reached_flags[sub_goal_reached] = 1

        ends[goal_reached & (self.sub_goal_reached_flags == 1)] = True
        results[goal_reached & (self.sub_goal_reached_flags == 1)] = 100

        return results, ends
