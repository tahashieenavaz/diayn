import torch
import warnings
from homa.rl import DiversityIsAllYouNeed
from utils import make_environment

warnings.filterwarnings("ignore")

algorithm = DiversityIsAllYouNeed(state_dimension=24, action_dimension=4)
env = make_environment("BipedalWalker-v3")

for episode in range(1000):
    done = False
    state, _ = env.reset()
    skill = algorithm.skill()

    while not done:
        state = torch.tensor(state).float().unsqueeze(0)
        skills = skill.repeat(state.size(0), 1)
        action, probability = algorithm.actor.action(state, skill)
        action = action.squeeze().detach().cpu().numpy()
        next_state, _, terminated, truncated, _ = env.step(action=action)
        done = truncated or terminated
        reward = algorithm.discriminator.reward(state, skill)
        next_state = torch.tensor(next_state).float()
        algorithm.buffer.record(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            probability=probability,
            termination=done,
        )
        state = next_state

    algorithm.train(skill=skill)
    algorithm.buffer.reset()

env.close()
