import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 加载 pgm 栅格地图 (白=可行 , 黑=障碍)
def load_pgm(path):
    img = Image.open(path)
    map_array = np.array(img)
    return (map_array == 255).astype(np.uint8)

# 判断是否能走
def is_valid(pos, grid_map):
    x, y = pos
    return 0 <= x < grid_map.shape[1] and 0 <= y < grid_map.shape[0] and grid_map[y, x] == 1

# DQN 推理路径
def infer_path(scripted_model, grid_map, start, goal, max_steps=100):
    path = [start]
    state = list(start) + list(goal)

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上下左右

    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = scripted_model(state_tensor).squeeze()
            sorted_actions = torch.argsort(q_values, descending=True).tolist()

        moved = False
        for action in sorted_actions:
            dx, dy = directions[action]
            next_pos = (path[-1][0] + dx, path[-1][1] + dy)

            if is_valid(next_pos, grid_map):
                path.append(next_pos)
                state = list(next_pos) + list(goal)
                moved = True
                break  # 找到合法动作就执行

        if not moved:
            print(f"无合法动作，停在：{path[-1]}")
            break

        if path[-1] == goal:
            break

    return path

# 在地图上画路径
def draw_path(grid_map, path):
    # img_ = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR)
    _, img = cv2.threshold(grid_map, 127, 255, cv2.THRESH_BINARY)

    for x, y in path:
        print(x, y)
        cv2.circle(img, (x, y), radius=1, color=(255, 0, 255), thickness=-1)
    plt.imshow(img)
    # plt.title("路径规划结果")
    # plt.show()

    plt.title("路径规划结果")
    plt.show(block=False)  # 非阻塞
    plt.pause(3)           # 显示3秒
    plt.close()


# === 主程序 ===
pgm_path = "C:/Users/20416/Desktop/rl_example/map/b.pgm"
model_path = "C:/Users/20416/Desktop/rl_example/model/dqn_path_planner.pt"
start = (1, 1)
goal = (300, 600)

image = cv2.imread(pgm_path)
cv2.imshow("window_name", image)
print(image.shape)
cv2.waitKey(0)

map_array = load_pgm(pgm_path)
model = torch.jit.load(model_path)  # TorchScript 模型加载
model.eval()

path = infer_path(model, map_array, start, goal)
draw_path(image, path)
