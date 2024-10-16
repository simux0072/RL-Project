import torch
from environment import EnvManager
from agent import Agent
from memory import Memory
import numpy
import time

input_size = 10 * 10
actions = 4
lr = 0.001
epochs = 16
minibatch_size = 32
epsilon = 0.2

gamma = 0.1
lambda_ = 0.95

memory_size = 100000

games_per_epochs = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory_buf = Memory(memory_size)
env_manager = EnvManager()
agent_obj = Agent(
    memory_buf, device, input_size, actions, lr, epochs, minibatch_size, epsilon
)


def get_color_coded_str(i):
    return "\u001b[4{}m  \u001b[0m".format(int(i) * 7 if i == 1 else int(i))


def print_a_ndarray(map, row_sep=" "):
    fmt_str = "\n".join([row_sep.join(["{}"] * map.shape[0])] * map.shape[1])
    print(fmt_str.format(*map.ravel()))


def create_everything():
    maze = numpy.ones((10, 10))
    maze[1:9, 1:9] = 0
    rand = numpy.random.randint(1, 9, 2)
    for _ in range(10):
        maze[rand[0], rand[1]] = 1
        map_modified = numpy.vectorize(get_color_coded_str)(maze)
        print_a_ndarray(map_modified)
        print("\033[A" * 11)
        maze[rand[0], rand[1]] = 0
        time.sleep(1)
        rand[0] += 0
        rand[1] += -1


def create_image(maze):
    map_modified = numpy.vectorize(get_color_coded_str)(maze)
    print_a_ndarray(map_modified)


epoch = 0

while True:
    game_id = 0
    for _ in range(games_per_epochs):
        idx = 0
        while True:
            idx += 1
            state = env_manager.get_state().copy()
            create_image(state)
            # state = torch.from_numpy(env_manager.get_state().copy()).unsqueeze(0)
            state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
            value, action, log_probs = agent_obj.get_action(state.to(device))
            reward, end = env_manager.make_move(action)
            if end or idx == 2000:
                idx = 0
                memory_buf.update_memory(
                    state,
                    action,
                    value.to(torch.device("cpu")),
                    log_probs.to(torch.device("cpu")),
                    True,
                    reward,
                )
                env_manager.reset_env()
                game_id += 1
                print(f"Game {game_id} finished.")
                print("\033[A" * 12)
                break

            memory_buf.update_memory(
                state,
                action,
                value.to(torch.device("cpu")),
                log_probs.to(torch.device("cpu")),
                end,
                reward,
            )
            print("\033[A" * 11)

    memory_buf.set_target_values(gamma, lambda_)
    actor_loss, critic_loss = agent_obj.train()
    memory_buf.restart()
    env_manager.reset_env()
    epoch += 1
    print(f"\nEpoch {epoch}: Actor Loss: {actor_loss}\nCritic Loss: {critic_loss}")
    print("\033[A" * 1)
