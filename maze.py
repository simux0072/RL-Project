import numpy

ACTIONS = numpy.array([[1, 0], [0, -1], [-1, 0], [0, 1]])  # Down, left, up, right.


class Maze:
    def __init__(self) -> None:
        self.restart()

    def restart(self):
        self.goals = []
        self.sub_goal_reached = False
        self.generate_objectives()

    def generate_objectives(self):
        positions = []
        for _ in range(3):
            positions.append(self.generate_coords(positions))
        self.player_pos = positions[0]
        self.goals = numpy.stack(positions[1:])

    def generate_coords(self, positions):
        goal = numpy.random.randint(1, 9, 2)
        if len(positions) != 0 and (goal == positions).all(1).any():
            return self.generate_coords(positions)
        else:
            return goal

    def move(self, action):
        # print(self.player_pos + ACTIONS[action])
        if (
            not (self.player_pos + ACTIONS[action] >= 9).any()
            and not (self.player_pos + ACTIONS[action] <= 0).any()
        ):
            self.player_pos += ACTIONS[action]
        return self.check_goal()

    def check_goal(self):
        if not self.sub_goal_reached and (self.goals[0] == self.player_pos).all():
            self.sub_goal_reached = True
            return 5, False  # Reward: int, end of episode: bool
        elif self.sub_goal_reached and (self.goals[1] == self.player_pos).all():
            return 10, True
        else:
            return 0, False
