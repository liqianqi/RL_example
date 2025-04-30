import gym
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

env = gym.make("CliffWalking-v0", render_mode=None)

n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))  # 初始化 Q 表
model = {}  # 模型 memory：s,a -> (r, s')

alpha = 0.1       # 学习率
gamma = 0.95      # 折扣因子
epsilon = 0.1     # 探索率
n_planning = 10   # 每步计划次数

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

n_episodes = 500

for episode in tqdm(range(n_episodes)):
    state, _ = env.reset()
    
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning 更新
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
        # 存储模型
        model[(state, action)] = (reward, next_state)

        # 模拟回放（Dyna部分）
        for _ in range(n_planning):
            s, a = random.choice(list(model.keys()))
            r, s_next = model[(s, a)]
            Q[s][a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
        
        state = next_state

# 展示策略
policy = np.array([np.argmax(Q[s]) for s in range(n_states)]).reshape((4, 12))
print("Learned Policy (0=↑, 1=↓, 2=←, 3=→):")
print(policy)
