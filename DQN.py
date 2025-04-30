import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import imageio.v2 as imageio
import random
import os


# === 1. 读取 PGM + YAML 地图 ===
def load_map(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    pgm_path = os.path.join(os.path.dirname(yaml_path), data['image'])
    resolution = data['resolution']
    origin = data['origin']
    negate = data.get('negate', 0)
    occ_thresh = data.get('occupied_thresh', 0.65)
    free_thresh = data.get('free_thresh', 0.196)

    img = imageio.imread(pgm_path).astype(np.float32) / 255.0
    if negate:
        img = 1.0 - img

    grid = np.zeros_like(img, dtype=np.uint8)
    grid[img < free_thresh] = 1  # Free
    grid[img > occ_thresh] = 0  # Occupied

    return grid, resolution, origin


# === 2. 自定义路径规划环境 ===
class GridEnv:
    def __init__(self, grid_map):
        self.grid = grid_map
        self.height, self.width = self.grid.shape
        self.goal = None
        self.state = None

    def reset(self, start, goal):
        self.state = start
        self.goal = goal
        return self._get_obs()

    def _get_obs(self):
        return np.array([*self.state, *self.goal], dtype=np.float32)

    def step(self, action):
        x, y = self.state
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        nx, ny = x + dx, y + dy

        # 边界与障碍判断
        if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 1:
            self.state = (nx, ny)

        done = self.state == self.goal
        reward = 1.0 if done else -0.1
        return self._get_obs(), reward, done, {}


# === 3. DQN 网络 === 
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 个动作
        )

    def forward(self, x):
        return self.fc(x)


# === 4. 训练主流程 ===
def train_dqn(yaml_path, episodes=1000):
    grid, res, origin = load_map(yaml_path)
    env = GridEnv(grid)
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    gamma = 0.99
    epsilon = 0.1

    for ep in range(episodes):
        while True:
            start = (random.randint(0, env.width - 1), random.randint(0, env.height - 1))
            goal = (random.randint(0, env.width - 1), random.randint(0, env.height - 1))
            if grid[start[1], start[0]] == 1 and grid[goal[1], goal[0]] == 1 and start != goal:
                break

        state = env.reset(start, goal)
        for t in range(200):
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    action = model(torch.tensor(state).unsqueeze(0)).argmax().item()

            next_state, reward, done, _ = env.step(action)

            with torch.no_grad():
                target_q = reward + gamma * model(torch.tensor(next_state).unsqueeze(0)).max().item() * (1 - int(done))

            q_pred = model(torch.tensor(state).unsqueeze(0))[0, action]
            loss = loss_fn(q_pred, torch.tensor(target_q))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} 完成 ✅")

    # torch.save(model, "dqn_path_planner.pt")
    dummy_input = torch.randn(1, 4)  # 改成你模型的实际输入 shape
    traced = torch.jit.trace(model, dummy_input)
    traced.save("dqn_path_planner.pt")

    print("🎉 模型已保存为 dqn_path_planner.pt")


# === 5. 启动训练 ===
if __name__ == "__main__":
    yaml_file_path = "C:/Users/20416/Desktop/rl_example/map/b.yaml"  # ⚠️ 替换成你的路径
    train_dqn(yaml_file_path, episodes=1000)
