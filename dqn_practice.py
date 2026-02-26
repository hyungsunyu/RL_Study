import gymnasium as gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        # transition: (s, a, r, s_prime, done_mask)
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

        # Fast conversion (fixes "Creating a tensor from a list..." warning)
        s_arr    = np.asarray(s_lst,       dtype=np.float32)
        a_arr    = np.asarray(a_lst,       dtype=np.int64)
        r_arr    = np.asarray(r_lst,       dtype=np.float32)
        sp_arr   = np.asarray(s_prime_lst, dtype=np.float32)
        done_arr = np.asarray(done_mask_lst, dtype=np.float32)

        # Keep on CPU here; move to GPU inside train() in one place
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
        # obs: torch.Tensor [4] or [1,4], already on device
        if random.random() < epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            q = self.forward(obs)
            return q.argmax().item()


def train(q, q_target, memory, optimizer):
    q.train()
    for _ in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # Move batch to GPU (core point)
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


def main():
    env = gym.make("CartPole-v1")  # 렌더 필요하면 render_mode="human" (추가 설치 필요할 수 있음)

    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    print_interval = 20
    score = 0.0

    # GPU가 제대로 붙었는지 1회 출력
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # 8% -> 1%
        s, _ = env.reset()

        done = False
        while not done:
            obs_t = torch.from_numpy(s).float().to(device)
            a = q.sample_action(obs_t, epsilon)

            # gymnasium: step returns (obs, reward, terminated, truncated, info)
            s_prime, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0
        # torch.save(q.state_dict(), "dqn_cartpole.pt")
        # print("Saved model to dqn_cartpole.pt")
    env.close()


if __name__ == "__main__":
    main()