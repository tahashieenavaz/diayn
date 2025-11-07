import torch
from homa.rl import DiversityIsAllYouNeed
from utils import make_environment

algorithm = DiversityIsAllYouNeed()
env = make_environment()

for episode in range(1000):
    done = False
    state, _ = env.reset()
    skill = algorithm.skill()

    while not done:
        state = torch.tensor(state).float().unsqueeze(0)
        skills = skill.repeat(state.size(0), 1)
        action, probability = algorithm.actor.action(state, skills)
        next_state, _, terminated, truncated, _ = env.step(action.item())
        done = truncated or terminated
        reward = algorithm.discriminator.reward(state, skill)
        algorithm.buffer.record(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            probability=probability,
        )
