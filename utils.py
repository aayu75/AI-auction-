import torch
import numpy as np

def simulate(env, model):
    state = env.reset()

    action = model(torch.tensor(state)).detach().numpy()
    _, reward, highest, second, efficiency = env.step(action)

    ai_payment = action[0] * highest + action[1] * second

    return ai_payment, highest, second, efficiency, action, env.bids
