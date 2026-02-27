import gymnasium as gym
import collections
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

CHECKPOINT_EPISODES = [1, 10, 100, 1000, 10000]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for (s, a, r, s_prime, done_mask) in mini_batch:
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_arr    = np.asarray(s_lst,         dtype=np.float32)
        a_arr    = np.asarray(a_lst,         dtype=np.int64)
        r_arr    = np.asarray(r_lst,         dtype=np.float32)
        sp_arr   = np.asarray(s_prime_lst,   dtype=np.float32)
        done_arr = np.asarray(done_mask_lst, dtype=np.float32)

        return (torch.from_numpy(s_arr),
                torch.from_numpy(a_arr),
                torch.from_numpy(r_arr),
                torch.from_numpy(sp_arr),
                torch.from_numpy(done_arr))

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            q = self.forward(obs)
            return q.argmax().item()


def train(q, q_target, memory, optimizer):
    q.train()
    for _ in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        s         = s.to(device)
        a         = a.to(device)
        r         = r.to(device)
        s_prime   = s_prime.to(device)
        done_mask = done_mask.to(device)

        q_out = q(s)
        q_a = q_out.gather(1, a)

        with torch.no_grad():
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def play_one_episode_render(q, env_render, title: str = ""):
    """greedy(ε=0)로 1판만 human 렌더 플레이"""
    q.eval()
    s, _ = env_render.reset()
    done = False
    ep_return = 0.0

    if title:
        print(title)

    while not done:
        env_render.render()
        obs_t = torch.from_numpy(s).float().to(device)
        a = q.sample_action(obs_t, epsilon=0.0)

        s, r, terminated, truncated, _ = env_render.step(a)
        done = terminated or truncated
        ep_return += r

    return ep_return


def main():
    env = gym.make("CartPole-v1")  # 학습용
    memory = ReplayBuffer()

    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    print_interval = 20
    score = 0.0

    # ✅ 체크포인트(메모리 저장)
    checkpoints = {}  # {episode: state_dict}

    for n_epi in range(1, 10001):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s, _ = env.reset()

        done = False
        while not done:
            obs_t = torch.from_numpy(s).float().to(device)
            a = q.sample_action(obs_t, epsilon)

            s_prime, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime
            score += r

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0:
            q_target.load_state_dict(q.state_dict())
            print(
                f"n_episode :{n_epi}, score : {score / print_interval:.1f}, "
                f"n_buffer : {memory.size()}, eps : {epsilon * 100:.1f}%"
            )
            score = 0.0

        # ✅ 에피소드 끝난 시점의 q 저장
        if n_epi in CHECKPOINT_EPISODES:
            checkpoints[n_epi] = deepcopy(q.state_dict())
            print(f"[CKPT] saved checkpoint at episode {n_epi}")

    env.close()

    # ===== 학습 종료 후 시각화 =====
    try:
        env_render = gym.make("CartPole-v1", render_mode="human")
    except Exception:
        print("[INFO] render_mode='human' 환경 생성 실패 (pygame 미설치 등).")
        return

    # 체크포인트 순서대로 1판씩 재생
    for ep in CHECKPOINT_EPISODES:
        if ep not in checkpoints:
            print(f"[WARN] checkpoint for episode {ep} not found.")
            continue

        q.load_state_dict(checkpoints[ep])
        ret = play_one_episode_render(q, env_render, title=f"\n=== Visualize checkpoint episode {ep} ===")
        print(f"[VIS] episode {ep}: greedy return = {ret:.1f}")

    env_render.close()


if __name__ == "__main__":
    main()