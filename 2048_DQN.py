import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------
# 2048 Environment (numpy)
# -----------------------------

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

def spawn_tile(board: np.ndarray, rng: random.Random) -> bool:
    empties = list(zip(*np.where(board == 0)))
    if not empties:
        return False
    r, c = rng.choice(empties)
    board[r, c] = 4 if rng.random() < 0.10 else 2
    return True

def compress_and_merge_row_left(row: np.ndarray):
    """Return (new_row, reward_gained, changed) for a single row moving LEFT."""
    nonzero = row[row != 0]
    merged = []
    reward = 0
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            val = int(nonzero[i] * 2)
            merged.append(val)
            reward += val
            i += 2
        else:
            merged.append(int(nonzero[i]))
            i += 1

    merged = np.array(merged, dtype=np.int64)
    new_row = np.zeros_like(row)
    new_row[: len(merged)] = merged
    changed = not np.array_equal(new_row, row)
    return new_row, reward, changed

def move_left(board: np.ndarray):
    new_board = board.copy()
    total_reward = 0
    changed_any = False
    for r in range(4):
        new_row, rwd, changed = compress_and_merge_row_left(new_board[r])
        new_board[r] = new_row
        total_reward += rwd
        changed_any = changed_any or changed
    return new_board, total_reward, changed_any

def move_right(board: np.ndarray):
    rev = np.fliplr(board)
    moved, rwd, changed = move_left(rev)
    return np.fliplr(moved), rwd, changed

def move_up(board: np.ndarray):
    trans = board.T
    moved, rwd, changed = move_left(trans)
    return moved.T, rwd, changed

def move_down(board: np.ndarray):
    trans = board.T
    moved, rwd, changed = move_right(trans)
    return moved.T, rwd, changed

def apply_action(board: np.ndarray, action: int):
    if action == 0:
        return move_up(board)
    if action == 1:
        return move_down(board)
    if action == 2:
        return move_left(board)
    if action == 3:
        return move_right(board)
    raise ValueError("Invalid action")

def has_any_valid_move(board: np.ndarray) -> bool:
    if (board == 0).any():
        return True
    for a in range(4):
        _, _, changed = apply_action(board, a)
        if changed:
            return True
    return False

def valid_actions(board: np.ndarray):
    acts = []
    for a in range(4):
        _, _, changed = apply_action(board, a)
        if changed:
            acts.append(a)
    return acts

class Env2048:
    def __init__(self, seed: int = 0, invalid_penalty: float = -1.0):
        self.rng = random.Random(seed)
        self.invalid_penalty = invalid_penalty
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.step_count = 0
        spawn_tile(self.board, self.rng)
        spawn_tile(self.board, self.rng)
        return self._get_obs()

    def _get_obs(self):
        # log2 encoding: empty=0, 2->1,4->2,...
        obs = np.zeros((16,), dtype=np.float32)
        flat = self.board.flatten()
        for i, v in enumerate(flat):
            obs[i] = 0.0 if v == 0 else float(math.log2(v))
        obs /= 16.0  # mild normalization
        return obs

    def step(self, action: int):
        self.step_count += 1
        nb, rwd, changed = apply_action(self.board, action)

        if not changed:
            reward = float(self.invalid_penalty)
            done = not has_any_valid_move(self.board)
            return self._get_obs(), reward, done, {"changed": False}

        self.board = nb
        spawn_tile(self.board, self.rng)
        self.score += int(rwd)
        reward = float(rwd)

        done = not has_any_valid_move(self.board)
        return self._get_obs(), reward, done, {"changed": True}


# -----------------------------
# DQN
# -----------------------------

@dataclass
class HParams:
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_limit: int = 100000
    batch_size: int = 256
    train_start: int = 5000
    train_iters_per_step: int = 1
    target_update_interval: int = 1000  # global steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200000       # linear decay
    max_episodes: int = 50000
    max_steps_per_episode: int = 5000
    render_every_steps: int = 10        # render update frequency

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def put(self, transition):
        self.buf.append(transition)

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, sp, dm, nmask = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1)
        sp = torch.tensor(np.array(sp), dtype=torch.float32)
        dm = torch.tensor(np.array(dm), dtype=torch.float32).unsqueeze(1)
        nmask = torch.tensor(np.array(nmask), dtype=torch.float32)  # (B,4)
        return s, a, r, sp, dm, nmask

    def __len__(self):
        return len(self.buf)

class QNet(nn.Module):
    def __init__(self, in_dim=16, hidden=256, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def linear_epsilon(step: int, hp: HParams):
    if step >= hp.eps_decay_steps:
        return hp.eps_end
    frac = step / hp.eps_decay_steps
    return hp.eps_start + frac * (hp.eps_end - hp.eps_start)

def masked_argmax(q_values: torch.Tensor, valid_mask: torch.Tensor):
    neg_inf = torch.tensor(-1e9, device=q_values.device, dtype=q_values.dtype)
    masked = torch.where(valid_mask > 0, q_values, neg_inf)
    return masked.argmax(dim=-1)

# -----------------------------
# Pygame Renderer (two boards)
# -----------------------------

def try_import_pygame():
    try:
        import pygame
        return pygame
    except Exception:
        return None

def tile_color(v: int):
    if v == 0:   return (205, 193, 180)
    if v == 2:   return (238, 228, 218)
    if v == 4:   return (237, 224, 200)
    if v == 8:   return (242, 177, 121)
    if v == 16:  return (245, 149, 99)
    if v == 32:  return (246, 124, 95)
    if v == 64:  return (246, 94, 59)
    if v == 128: return (237, 207, 114)
    if v == 256: return (237, 204, 97)
    if v == 512: return (237, 200, 80)
    if v == 1024:return (237, 197, 63)
    if v == 2048:return (237, 194, 46)
    return (60, 58, 50)

def text_color(v: int):
    return (119, 110, 101) if v in (2, 4) else (249, 246, 242)

class Renderer2048:
    def __init__(self, pygame):
        self.pygame = pygame
        pygame.init()
        self.W = 900
        self.H = 520
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("2048 DQN - Current vs Best")
        self.font_mid = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_tile = pygame.font.SysFont("consolas", 34, bold=True)

        self.bg = (250, 248, 239)
        self.panel_bg = (187, 173, 160)
        self.running = True

        # layout
        self.pad = 16
        self.tile_gap = 10
        self.board_size = 360
        self.left_x = 60
        self.right_x = 480
        self.board_y = 130

    def handle_events(self):
        for e in self.pygame.event.get():
            if e.type == self.pygame.QUIT:
                self.running = False

    def draw_board(self, board: np.ndarray, x0: int, y0: int):
        pg = self.pygame
        pg.draw.rect(self.screen, self.panel_bg, (x0, y0, self.board_size, self.board_size), border_radius=12)

        tile_w = (self.board_size - 2*self.pad - 3*self.tile_gap) // 4
        for r in range(4):
            for c in range(4):
                v = int(board[r, c])
                x = x0 + self.pad + c*(tile_w + self.tile_gap)
                y = y0 + self.pad + r*(tile_w + self.tile_gap)
                pg.draw.rect(self.screen, tile_color(v), (x, y, tile_w, tile_w), border_radius=10)

                if v != 0:
                    s = str(v)
                    surf = self.font_tile.render(s, True, text_color(v))
                    rect = surf.get_rect(center=(x + tile_w//2, y + tile_w//2))
                    self.screen.blit(surf, rect)

    def draw_header_block(self, *, x: int, score: int, episode: int, best: int, step: int):
        pg = self.pygame
        color = (90, 80, 70)

        t1 = self.font_mid.render(f"score: {score}", True, color)
        t2 = self.font_mid.render(f"episode: {episode}", True, color)
        t3 = self.font_mid.render(f"best: {best}", True, color)
        t4 = self.font_mid.render(f"step: {step}", True, color)

        self.screen.blit(t1, (x, 20))
        self.screen.blit(t2, (x + 260, 20))
        self.screen.blit(t3, (x, 55))
        self.screen.blit(t4, (x + 260, 55))

    def render(
        self,
        left_board: np.ndarray,
        right_board: np.ndarray,
        *,
        left_score: int,
        left_episode: int,
        left_step: int,
        best_score: int,
        best_episode: int,
        best_step: int
    ):
        self.handle_events()
        if not self.running:
            return

        self.screen.fill(self.bg)

        # LEFT header: current info
        self.draw_header_block(
            x=self.left_x,
            score=left_score,
            episode=left_episode,
            best=best_score,
            step=left_step
        )

        # RIGHT header: best snapshot info (✅ best_episode 표시)
        self.draw_header_block(
            x=self.right_x,
            score=best_score,
            episode=best_episode,
            best=best_score,
            step=best_step
        )

        self.draw_board(left_board, self.left_x, self.board_y)
        self.draw_board(right_board, self.right_x, self.board_y)

        self.pygame.display.flip()

# -----------------------------
# Training
# -----------------------------

def train_2048_with_visual():
    hp = HParams()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = QNet().to(device)
    q_t = QNet().to(device)
    q_t.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=hp.lr)
    buf = ReplayBuffer(hp.buffer_limit)

    env = Env2048(seed=0, invalid_penalty=-1.0)

    pygame = try_import_pygame()
    renderer = Renderer2048(pygame) if pygame is not None else None
    if renderer is None:
        print("[INFO] pygame이 없어서 실시간 시각화는 비활성화됩니다.")

    global_step = 0

    # ✅ best snapshot tracking (only update when best improves)
    best_score = -1
    best_board = np.zeros((4, 4), dtype=np.int64)
    best_step = 0
    best_episode = 0

    for ep in range(1, hp.max_episodes + 1):
        obs = env.reset()
        done = False
        ep_steps = 0

        while not done and ep_steps < hp.max_steps_per_episode:
            ep_steps += 1
            global_step += 1

            # valid mask for current state
            vacts = valid_actions(env.board)
            if not vacts:
                # no valid actions -> terminal
                done = True
                break

            vmask = np.zeros((4,), dtype=np.float32)
            for a in vacts:
                vmask[a] = 1.0

            eps = linear_epsilon(global_step, hp)

            # epsilon-greedy with valid-action set
            if random.random() < eps:
                action = random.choice(vacts)
            else:
                with torch.no_grad():
                    s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    qv = q(s_t).squeeze(0)  # (4,)
                    vm_t = torch.tensor(vmask, dtype=torch.float32, device=device)
                    action = int(masked_argmax(qv.unsqueeze(0), vm_t.unsqueeze(0)).item())

            obs2, reward, done, info = env.step(action)

            # next valid mask (for masked max in target)
            nvacts = valid_actions(env.board)
            nmask = np.zeros((4,), dtype=np.float32)
            for a in nvacts:
                nmask[a] = 1.0

            buf.put((obs, action, reward, obs2, 0.0 if done else 1.0, nmask))
            obs = obs2

            # ✅ update best snapshot ONLY when score improves
            if env.score > best_score:
                best_score = env.score
                best_board = env.board.copy()
                best_step = env.step_count
                best_episode = ep

            # visualize occasionally
            if renderer is not None and (global_step % hp.render_every_steps == 0):
                renderer.render(
                    left_board=env.board,
                    right_board=best_board,
                    left_score=env.score,
                    left_episode=ep,
                    left_step=env.step_count,
                    best_score=best_score,
                    best_episode=best_episode,   # ✅ 고정된 best episode
                    best_step=best_step
                )
                if not renderer.running:
                    print("[INFO] 창이 닫혀서 종료합니다.")
                    return

            # training
            if len(buf) >= hp.train_start:
                for _ in range(hp.train_iters_per_step):
                    s, a, r, sp, dm, nvm = buf.sample(hp.batch_size)
                    s = s.to(device)
                    a = a.to(device)
                    r = r.to(device)
                    sp = sp.to(device)
                    dm = dm.to(device)
                    nvm = nvm.to(device)

                    q_sa = q(s).gather(1, a)

                    with torch.no_grad():
                        q_next = q_t(sp)  # (B,4)
                        neg_inf = torch.tensor(-1e9, device=device, dtype=q_next.dtype)
                        q_next_masked = torch.where(nvm > 0, q_next, neg_inf)
                        max_next = q_next_masked.max(dim=1, keepdim=True)[0]
                        target = r + hp.gamma * max_next * dm

                    loss = F.smooth_l1_loss(q_sa, target)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    opt.step()

            # target net update
            if global_step % hp.target_update_interval == 0:
                q_t.load_state_dict(q.state_dict())

        if ep % 100 == 0:
            print(f"[ep {ep}] score={env.score} steps={ep_steps} eps={linear_epsilon(global_step, hp):.3f} best={best_score} (best_ep={best_episode})")

        if renderer is not None and not renderer.running:
            break

    print("Training finished.")
    if pygame is not None:
        pygame.quit()

if __name__ == "__main__":
    train_2048_with_visual()