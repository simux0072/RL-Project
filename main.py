import torch
from environments import EnvManager
from agent import Agent
from memories import Memory
import numpy
import wandb


def get_color_coded_str(i):
    return "\u001b[4{}m  \u001b[0m".format(int(i) * 7 if i == 1 else int(i))


def print_a_ndarray(map, row_sep=" "):
    fmt_str = "\n".join([row_sep.join(["{}"] * map.shape[0])] * map.shape[1])
    print(fmt_str.format(*map.ravel()))


def create_image(maze):
    map_modified = numpy.vectorize(get_color_coded_str)(maze)
    print_a_ndarray(map_modified)


def play_environment(mem_buf, config):
    idx = 0
    env_finished = 0
    rewards_int_arr = []
    avg_rewards = 0
    while True:
        rewards_int = 0
        state = env_manager.env.mazes.copy()
        masks = torch.from_numpy(env_manager.env.create_mask()).to(config.device)
        values, actions, logits = agent_obj.action(
            torch.from_numpy(state).unsqueeze(dim=1).to(config.device), masks
        )
        actions = numpy.array(actions, dtype=int)
        rewards_ext, ends = env_manager.make_move(actions)
        if config.exploration_algorithm == "RND":
            rewards_int = agent_obj.get_intrinsic_reward(
                torch.from_numpy(state).unsqueeze(dim=1).to(config.device), values
            )
            rewards_int_arr.append(rewards_int.flatten())
        rewards = rewards_ext + config.alpha * rewards_int
        if ends.all() == True or idx == config.max_step_size - 1:
            env_finished += ends.sum()
            ends[:] = True
            has_end, end_idx = mem_buf.add_new_steps(
                state, actions, logits, rewards, ends, values, masks
            )
            env_manager.reset_env()
            print(
                f"Average rewards: {round(numpy.concatenate(mem_buf.rewards).sum()/config.num_games, 5)}"
            )
            print(f"Env finished: {env_finished}")
            avg_rewards = numpy.concatenate(mem_buf.rewards).sum() / config.num_games
            break

        has_end, end_idx = mem_buf.add_new_steps(
            state, actions, logits, rewards, ends, values, masks
        )

        if has_end:
            env_manager.remove_ended_games(end_idx)
            env_finished += end_idx.shape[0]
        idx += 1
    if config.exploration_algorithm == "RND":
        print(f"Intrinsic rewards: {numpy.concatenate(rewards_int_arr).mean()}")
        return (
            avg_rewards - numpy.concatenate(rewards_int_arr).mean(),
            numpy.array(rewards_int).mean(),
        )
    return avg_rewards, None


def eval_agent(agent_eval, config):
    idx = 0
    env_finished = 0
    eval_env = EnvManager(config.dims, 1, config.use_maze, config.use_static_goals)
    probs_arr = []
    actions_arr = []
    masks_arr = []
    while True:
        state = eval_env.env.mazes.copy()
        create_image(state[0])
        masks = torch.from_numpy(eval_env.env.create_mask()).to(config.device)
        actions, probs = agent_eval(
            torch.from_numpy(state).unsqueeze(dim=1).to(config.device), masks
        )
        probs_arr.append(probs)
        actions_arr.append(actions)
        masks_arr.append(masks)
        actions = numpy.array(actions, dtype=int)
        _, ends = eval_env.make_move(actions)
        if ends == True or idx == config.max_step_size - 1:
            env_finished = 1 if ends == 1 else 0
            break
        print("\033[A" * 11)
        idx += 1

    print(f"ends: {ends}")
    print(
        "Agent finished the environmet"
        if env_finished == 1
        else "Agent did not finish the environment"
    )
    print(f"Took: {idx + 1} steps")
    return (
        idx + 1,
        torch.cat(probs_arr),
        torch.tensor(actions_arr),
        torch.cat(masks_arr).to(torch.uint8),
        False if env_finished == 1 and idx + 1 == 15 else False,
    )


if __name__ == "__main__":
    wandb.login()

    num_experiments = 8

    config = dict(
        architecture="PPO",
        exploration_algorithm="AGAC",
        dims=[10, 10],
        actions=4,
        use_maze=True,
        use_static_goals=False,
        lr=0.0001,
        adver_lr=0.00003,
        epochs=4,
        minibatch_size=8,
        epsilon=0.2,
        gamma=0.95,
        lambda_=0.95,
        num_games=4,
        max_step_size=500,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        alpha=0.0001,
        c=10,
        annealing_rate=-4e-3,
    )

    for exp_num in range(num_experiments):
        run = wandb.init(project="University-RL_Project", config=config)

        memory_buf = Memory(
            wandb.config.num_games, wandb.config.max_step_size, wandb.config.dims
        )
        env_manager = EnvManager(
            wandb.config.dims,
            wandb.config.num_games,
            wandb.config.use_maze,
            wandb.config.use_static_goals,
        )
        agent_obj = Agent(
            wandb.config.device,
            wandb.config.dims[0] * wandb.config.dims[1],
            wandb.config.actions,
            wandb.config.lr,
            wandb.config.adver_lr,
            wandb.config.epochs,
            wandb.config.minibatch_size,
            wandb.config.epsilon,
            wandb.config.c,
            wandb.config.annealing_rate,
        )

        wandb.watch(agent_obj.actor)

        epoch = 1
        agent_reached_end = 0
        steps = 0
        while True:
            avg_rewards, RND_rewards = play_environment(memory_buf, wandb.config)
            memory_buf.set_target_values(wandb.config.gamma, wandb.config.lambda_)

            states, actions, logits, advantages, returns, masks = memory_buf.get_memory(
                wandb.config.device
            )
            steps += actions.size(0)
            if wandb.config.exploration_algorithm == "None":
                actor_loss, critic_loss, entropy = agent_obj.train(
                    states, actions, logits, advantages, returns, masks
                )
                print(
                    f"\nEnv Steps: {steps}\nEpoch {epoch}: Actor Loss: {actor_loss}\nCritic Loss: {critic_loss}\nEntropy: {entropy}"
                )
            elif wandb.config.exploration_algorithm == "RND":
                actor_loss, critic_loss, entropy = agent_obj.train(
                    states, actions, logits, advantages, returns, masks
                )
                print(
                    f"\nEnv Steps: {steps}\nEpoch {epoch}: Actor Loss: {actor_loss}\nCritic Loss: {critic_loss}\nEntropy: {entropy}"
                )
            elif wandb.config.exploration_algorithm == "AGAC":
                actor_loss, critic_loss, adver_loss = agent_obj.train_agac(
                    states,
                    actions,
                    logits,
                    advantages,
                    returns,
                    masks,
                )
                print(
                    f"\nEnv Steps: {steps}\nEpoch {epoch}: Actor Loss: {actor_loss}\nCritic Loss: {critic_loss}\nAdverserial Loss: {adver_loss}"
                )
            (
                agent_finish_idx,
                probs_tensor,
                actions_tensor,
                masks_tensor,
                reached_end,
            ) = eval_agent(agent_obj.eval_action, wandb.config)

            memory_buf.restart(
                wandb.config.num_games, wandb.config.max_step_size, wandb.config.dims
            )
            if wandb.config.exploration_algorithm == "None":
                wandb.log(
                    {
                        "Environment Steps": steps,
                        "epoch": epoch,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "entropy": entropy,
                        "Finish_Index": agent_finish_idx,
                        "Probabilities": probs_tensor,
                        "Actions": actions_tensor,
                        "Masks": masks_tensor,
                        "Average_Rewards": avg_rewards,
                    }
                )
            elif wandb.config.exploration_algorithm == "AGAC":
                wandb.log(
                    {
                        "Environment Steps": steps,
                        "epoch": epoch,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "adverserial_loss": adver_loss,
                        "Finish_Index": agent_finish_idx,
                        "Probabilities": probs_tensor,
                        "Actions": actions_tensor,
                        "Masks": masks_tensor,
                        "Average_Rewards": avg_rewards,
                        "AGAC KL Divergence": agent_obj.kl_div,
                        "AGAC Return Difference": agent_obj.returns_diff,
                        "AGAC Advantage Difference": agent_obj.adv_diff,
                        "AGAC Constant": agent_obj.c,
                    }
                )
            elif wandb.config.exploration_algorithm == "RND":
                wandb.log(
                    {
                        "Environment Steps": steps,
                        "epoch": epoch,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "entropy": entropy,
                        "Finish_Index": agent_finish_idx,
                        "Probabilities": probs_tensor,
                        "Actions": actions_tensor,
                        "Masks": masks_tensor,
                        "Average_Rewards": avg_rewards,
                        "RND Rewards": RND_rewards,
                    }
                )

            if reached_end:
                agent_reached_end += 1
            else:
                agent_reached_end = 0

            if agent_reached_end == 5 or steps == 2e9:
                print(
                    f"\n\nAgent has consistently reached end!\nEnd of experiment: {exp_num + 1}\nStarting new experiment"
                )
                run.finish()
                break

            env_manager.reset_env()
            epoch += 1
    print(f"Finished all experiments!")
