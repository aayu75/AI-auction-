import torch
import torch.optim as optim
from model import PolicyNet

def train_model(env, episodes=300):
    model = PolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)

        action = model(state)
        action_np = action.detach().numpy()

        _, reward, _, _, _ = env.step(action_np)

        loss = -reward * torch.mean(torch.log(action + 1e-6))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
