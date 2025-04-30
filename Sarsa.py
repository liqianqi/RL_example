# import numpy as np
# import gym

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # 超参数
# alpha = 0.1       # 学习率
# gamma = 0.99      # 折扣因子
# epsilon = 0.1     # 探索率
# episodes = 500    # 训练回合数

# # 创建环境
# env = gym.make('CliffWalking-v0', render_mode="ansi")
# n_states = env.observation_space.n
# n_actions = env.action_space.n

# # 初始化 Q表
# Q = np.zeros((n_states, n_actions))

# # 训练
# for episode in range(episodes):
#     state, _ = env.reset()
#     state = int(state)

#     done = False
#     while not done:
#         # epsilon-贪婪策略
#         if np.random.rand() < epsilon:
#             action = env.action_space.sample()
#         else:
#             action = np.argmax(Q[state])

#         next_state, reward, terminated, truncated, _ = env.step(action)
#         next_state = int(next_state)
#         done = terminated or truncated

#         # Q-learning 更新
#         best_next_action = np.argmax(Q[next_state])
#         Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

#         state = next_state

# print("训练完毕！✅")

# # 测试：展示学到的最优路径
# state, _ = env.reset()
# state = int(state)
# env.render()

# done = False
# while not done:
#     action = np.argmax(Q[state])
#     next_state, reward, terminated, truncated, _ = env.step(action)
#     next_state = int(next_state)
#     done = terminated or truncated

#     print(env.render())

#     # env.render()

#     state = next_state
# env.close()

import numpy as np
import random
from tqdm import trange

# 定义 CliffWalking 环境（4行12列）
class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.P = self._create_transition_matrix()

    def _create_transition_matrix(self):
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # 上下左右

        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    state = i * self.ncol + j
                    if i == self.nrow - 1 and j > 0:
                        # 悬崖 + 终点
                        P[state][a] = [(1.0, state, 0, True)]
                        continue

                    dx, dy = directions[a]
                    next_x = min(self.ncol - 1, max(0, j + dx))
                    next_y = min(self.nrow - 1, max(0, i + dy))
                    next_state = next_y * self.ncol + next_x

                    reward = -1
                    done = False
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 掉悬崖
                            reward = -100

                    P[state][a] = [(1.0, next_state, reward, done)]
        return P

# 初始化
env = CliffWalkingEnv()
n_states = env.nrow * env.ncol
n_actions = 4
Q = np.zeros((n_states, n_actions))

# 超参数
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 500

# Q-learning 主循环
for _ in trange(episodes):
    state = env.ncol * (env.nrow - 1)  # 起点 (3, 0)
    done = False

    while not done:
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[state])

        _, next_state, reward, done = env.P[state][action][0]
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state = next_state

print("训练完毕！✅")

# 显示策略图
policy = np.chararray((env.nrow, env.ncol), unicode=True)
actions = ['↑', '↓', '←', '→']
for s in range(n_states):
    x = s % env.ncol
    y = s // env.ncol
    if y == env.nrow - 1 and x > 0:
        policy[y][x] = '⛔' if x != env.ncol - 1 else '🏁'
    else:
        policy[y][x] = actions[np.argmax(Q[s])]
print("最优策略图：")
print(policy)
