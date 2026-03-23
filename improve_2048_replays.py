import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------------------------------------------
# 2048 core helpers (kept compatible with train_2048.py)
# -------------------------------------------------

def spawn_tile(board: np.ndarray, rng: random.Random) -> bool:
    empties = list(zip(*np.where(board == 0)))
    if not empties:
        return False
    r, c = rng.choice(empties)
    board[r, c] = 4 if rng.random() < 0.10 else 2
    return True


def compress_and_merge_row_left(row: np.ndarray):
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
    out = np.zeros_like(row)
    out[: len(merged)] = merged
    return out, reward


def move_left(board: np.ndarray):
    new = board.copy()
    total_reward = 0
    changed = False
    for r in range(4):
        row = new[r, :]
        new_row, rew = compress_and_merge_row_left(row)
        if not np.array_equal(row, new_row):
            changed = True
        new[r, :] = new_row
        total_reward += rew
    return new, total_reward, changed


def move_right(board: np.ndarray):
    flipped = np.fliplr(board)
    new, rew, changed = move_left(flipped)
    return np.fliplr(new), rew, changed


def move_up(board: np.ndarray):
    trans = board.T
    new, rew, changed = move_left(trans)
    return new.T, rew, changed


def move_down(board: np.ndarray):
    trans = board.T
    new, rew, changed = move_right(trans)
    return new.T, rew, changed


def apply_action(board: np.ndarray, action: int):
    # 0: up, 1: down, 2: left, 3: right
    if action == 0:
        return move_up(board)
    if action == 1:
        return move_down(board)
    if action == 2:
        return move_left(board)
    if action == 3:
        return move_right(board)
    raise ValueError(f"invalid action: {action}")


def valid_actions(board: np.ndarray) -> List[int]:
    acts = []
    for a in range(4):
        _, _, changed = apply_action(board, a)
        if changed:
            acts.append(a)
    return acts


def has_any_valid_move(board: np.ndarray) -> bool:
    return len(valid_actions(board)) > 0


# -------------------------------------------------
# Replay IO (same base format as train_2048.py)
# -------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_npz_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    files.sort()
    return [os.path.join(folder, f) for f in files]


def load_npz(path: str):
    data = np.load(path, allow_pickle=False)
    try:
        boards = data["boards"].astype(np.uint32)
        actions = data["actions"].astype(np.uint8)
        meta = json.loads(str(data["meta"]))
    finally:
        data.close()
    return boards, actions, meta


def read_npz_meta_only(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    try:
        meta = json.loads(str(data["meta"]))
    finally:
        data.close()
    return meta


def sanitize_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    out = []
    for ch in stem:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120]


def save_replay_npz(
    save_dir: str,
    *,
    file_tag: str,
    score: int,
    max_tile: int,
    steps: int,
    boards_u32: np.ndarray,
    actions_u8: np.ndarray,
    extra_meta: Optional[Dict] = None,
) -> str:
    ensure_dir(save_dir)
    ts = int(time.time()%100)
    fname = f"best_tile{max_tile:06d}_score{score:09d}_ep{ts:06d}.npz"
    path = os.path.join(save_dir, fname)

    meta = {
        "episode": -1,
        "score": int(score),
        "max_tile": int(max_tile),
        "steps": int(steps),
        "boards_len": int(boards_u32.shape[0]),
        "actions_len": int(actions_u8.shape[0]),
        "saved_at_unix": ts,
        "generator": "rollback_mc_search_v1",
    }
    if extra_meta:
        meta.update(extra_meta)

    np.savez_compressed(
        path,
        boards=boards_u32,
        actions=actions_u8,
        meta=json.dumps(meta, ensure_ascii=False),
    )
    return path


# -------------------------------------------------
# Board evaluation heuristics
# -------------------------------------------------

def log2_board(board: np.ndarray) -> np.ndarray:
    out = np.zeros_like(board, dtype=np.float32)
    nonzero = board > 0
    out[nonzero] = np.log2(board[nonzero]).astype(np.float32)
    return out


def count_empty(board: np.ndarray) -> int:
    return int(np.sum(board == 0))


def max_tile(board: np.ndarray) -> int:
    return int(np.max(board))


def monotonicity_score(board: np.ndarray) -> float:
    b = log2_board(board)

    def line_mono(line: np.ndarray) -> float:
        inc = 0.0
        dec = 0.0
        for i in range(3):
            a = float(line[i])
            c = float(line[i + 1])
            if a <= c:
                inc += c - a
            if a >= c:
                dec += a - c
        return -min(inc, dec)

    score = 0.0
    for r in range(4):
        score += line_mono(b[r, :])
    for c in range(4):
        score += line_mono(b[:, c])
    return score


def smoothness_score(board: np.ndarray) -> float:
    b = log2_board(board)
    score = 0.0
    for r in range(4):
        for c in range(4):
            if b[r, c] <= 0:
                continue
            if r + 1 < 4 and b[r + 1, c] > 0:
                score -= abs(float(b[r, c] - b[r + 1, c]))
            if c + 1 < 4 and b[r, c + 1] > 0:
                score -= abs(float(b[r, c] - b[r, c + 1]))
    return score


def merge_potential(board: np.ndarray) -> float:
    score = 0.0
    for r in range(4):
        for c in range(4):
            v = int(board[r, c])
            if v == 0:
                continue
            if r + 1 < 4 and board[r + 1, c] == v:
                score += 1.0
            if c + 1 < 4 and board[r, c + 1] == v:
                score += 1.0
    return score


def corner_max_bonus(board: np.ndarray) -> float:
    m = max_tile(board)
    corners = [int(board[0, 0]), int(board[0, 3]), int(board[3, 0]), int(board[3, 3])]
    return 1.0 if m in corners else 0.0


def snake_score(board: np.ndarray) -> float:
    b = log2_board(board)
    snakes = [
        np.array([
            [15, 14, 13, 12],
            [8, 9, 10, 11],
            [7, 6, 5, 4],
            [0, 1, 2, 3],
        ], dtype=np.float32),
        np.array([
            [12, 13, 14, 15],
            [11, 10, 9, 8],
            [4, 5, 6, 7],
            [3, 2, 1, 0],
        ], dtype=np.float32),
        np.array([
            [3, 4, 11, 12],
            [2, 5, 10, 13],
            [1, 6, 9, 14],
            [0, 7, 8, 15],
        ], dtype=np.float32),
        np.array([
            [12, 11, 4, 3],
            [13, 10, 5, 2],
            [14, 9, 6, 1],
            [15, 8, 7, 0],
        ], dtype=np.float32),
    ]
    vals = [float(np.sum(b * w)) for w in snakes]
    return max(vals)


def board_value(board: np.ndarray) -> float:
    empty = count_empty(board)
    mt = max_tile(board)
    lg = math.log(mt, 2) if mt > 0 else 0.0
    return (
        300.0 * empty
        + 40.0 * lg
        + 120.0 * corner_max_bonus(board)
        + 14.0 * monotonicity_score(board)
        + 6.0 * smoothness_score(board)
        + 20.0 * merge_potential(board)
        + 3.5 * snake_score(board)
    )


# -------------------------------------------------
# Replay accounting helpers
# -------------------------------------------------

def cumulative_scores_from_replay(boards: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if boards.ndim != 3 or boards.shape[1:] != (4, 4):
        raise ValueError("boards must have shape (T, 4, 4)")
    if len(actions) != len(boards) - 1:
        raise ValueError("actions length must be boards length - 1")

    out = np.zeros((len(boards),), dtype=np.int64)
    score = 0
    for t, a in enumerate(actions):
        _, rew, changed = apply_action(boards[t].astype(np.int64), int(a))
        if not changed:
            raise ValueError(f"invalid stored action at t={t}: {a}")
        score += int(rew)
        out[t + 1] = score
    return out


def replay_stats(boards: np.ndarray, actions: np.ndarray, meta: Optional[dict] = None) -> Dict:
    cum = cumulative_scores_from_replay(boards.astype(np.int64), actions.astype(np.int64))
    score = int(cum[-1]) if len(cum) else 0
    mt = int(np.max(boards[-1])) if len(boards) else 0
    return {
        "score": score,
        "max_tile": mt,
        "steps": int(len(actions)),
        "meta_score": None if meta is None else meta.get("score"),
        "meta_max_tile": None if meta is None else meta.get("max_tile"),
    }


def objective_tuple(score: int, max_tile_value: int, steps: int, target_tile: int) -> Tuple[int, int, int, int]:
    reached = 1 if max_tile_value >= target_tile else 0
    return (reached, int(max_tile_value), int(score), -int(steps))


def is_better_candidate(
    score_a: int,
    max_tile_a: int,
    steps_a: int,
    score_b: int,
    max_tile_b: int,
    steps_b: int,
    target_tile: int,
) -> bool:
    return objective_tuple(score_a, max_tile_a, steps_a, target_tile) > objective_tuple(
        score_b, max_tile_b, steps_b, target_tile
    )


# -------------------------------------------------
# Monte Carlo action selection
# -------------------------------------------------

@dataclass
class SearchConfig:
    input_dir: str = "replays_best"
    output_dir: str = "replays_improved"
    summary_json: str = "replays_improved/summary.json"

    max_files: int = 64
    min_source_max_tile: int = 1024
    max_source_max_tile: int = 4096
    skip_if_already_target: bool = True

    target_tile: int = 16384
    max_rollback: int = 32
    attempts_per_pivot: int = 8
    outer_passes: int = 2
    max_steps_per_attempt: int = 1500

    preselect_top_actions: int = 2
    rollouts_per_action: int = 0
    rollout_depth: int = 12
    spawn_eval_samples: int = 4
    exploration_eps: float = 0.06
    top_rollout_bonus: float = 0.20

    keep_best_outputs: int = 128
    stop_after_first_target: bool = False
    random_seed: int = 20260320
    verbose: bool = True


def sample_post_spawn_states(board_after_move: np.ndarray, rng: random.Random, samples: int) -> List[np.ndarray]:
    empties = list(zip(*np.where(board_after_move == 0)))
    if not empties:
        return [board_after_move.copy()]

    total_exact = len(empties) * 2
    states: List[np.ndarray] = []

    if total_exact <= max(2, samples):
        for r, c in empties:
            b2 = board_after_move.copy()
            b2[r, c] = 2
            states.append(b2)
            b4 = board_after_move.copy()
            b4[r, c] = 4
            states.append(b4)
        return states

    for _ in range(samples):
        b = board_after_move.copy()
        r, c = rng.choice(empties)
        b[r, c] = 4 if rng.random() < 0.10 else 2
        states.append(b)
    return states


def greedy_one_step_action(board: np.ndarray, rng: random.Random) -> int:
    acts = valid_actions(board)
    if not acts:
        return -1

    best_val = None
    best_actions: List[int] = []
    for a in acts:
        b1, rew, changed = apply_action(board, a)
        if not changed:
            continue
        v = 1000.0 * float(rew) + board_value(b1)
        if best_val is None or v > best_val + 1e-9:
            best_val = v
            best_actions = [a]
        elif abs(v - best_val) <= 1e-9:
            best_actions.append(a)
    return rng.choice(best_actions)


def short_rollout_tail(board: np.ndarray, rng: random.Random, cfg: SearchConfig) -> float:
    b = board.copy()
    total = 0.0
    for _ in range(cfg.rollout_depth):
        acts = valid_actions(b)
        if not acts:
            break
        if rng.random() < cfg.exploration_eps:
            a = rng.choice(acts)
        else:
            a = greedy_one_step_action(b, rng)
            if a < 0:
                break
        b2, rew, changed = apply_action(b, a)
        if not changed:
            break
        spawn_tile(b2, rng)
        total += 1000.0 * float(rew)
        b = b2
    total += board_value(b)
    return total


def action_value(board: np.ndarray, action: int, rng: random.Random, cfg: SearchConfig) -> float:
    b1, rew, changed = apply_action(board, action)
    if not changed:
        return float("-inf")

    base = 1000.0 * float(rew)
    post_states = sample_post_spawn_states(b1, rng, cfg.spawn_eval_samples)
    post_val = 0.0
    for st in post_states:
        post_val += board_value(st)
    post_val /= float(len(post_states))

    if cfg.rollouts_per_action <= 0:
        return base + post_val

    rollout_vals = []
    for _ in range(cfg.rollouts_per_action):
        local_rng = random.Random(rng.randrange(1 << 62))
        b2 = b1.copy()
        spawn_tile(b2, local_rng)
        rollout_vals.append(base + short_rollout_tail(b2, local_rng, cfg))

    mean_rollout = float(np.mean(np.array(rollout_vals, dtype=np.float64)))
    max_rollout = float(np.max(np.array(rollout_vals, dtype=np.float64)))
    return 0.50 * (base + post_val) + 0.50 * mean_rollout + cfg.top_rollout_bonus * max_rollout


def select_action(board: np.ndarray, rng: random.Random, cfg: SearchConfig) -> int:
    acts = valid_actions(board)
    if not acts:
        return -1

    quick_scores = []
    for a in acts:
        b1, rew, changed = apply_action(board, a)
        if not changed:
            continue
        quick_scores.append((1000.0 * float(rew) + board_value(b1), a))

    quick_scores.sort(reverse=True)
    candidate_actions = [a for _, a in quick_scores[: max(1, cfg.preselect_top_actions)]]
    best_val = None
    best_actions: List[int] = []
    for a in candidate_actions:
        v = action_value(board, a, rng, cfg)
        if best_val is None or v > best_val + 1e-9:
            best_val = v
            best_actions = [a]
        elif abs(v - best_val) <= 1e-9:
            best_actions.append(a)

    return rng.choice(best_actions)


# -------------------------------------------------
# Rollback search
# -------------------------------------------------

@dataclass
class SimulationResult:
    boards: np.ndarray
    actions: np.ndarray
    score: int
    max_tile: int
    steps: int
    heuristic_value: float
    rollback_used: int
    pass_index: int
    source_file: str


def rollout_from_board(
    start_board: np.ndarray,
    start_score: int,
    rng: random.Random,
    cfg: SearchConfig,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    board = start_board.astype(np.int64).copy()
    score = int(start_score)
    boards = [board.copy().astype(np.uint32)]
    actions: List[int] = []

    for _ in range(cfg.max_steps_per_attempt):
        acts = valid_actions(board)
        if not acts:
            break

        action = select_action(board, rng, cfg)
        if action < 0:
            break

        new_board, rew, changed = apply_action(board, action)
        if not changed:
            break
        spawn_tile(new_board, rng)
        score += int(rew)
        board = new_board
        actions.append(int(action))
        boards.append(board.copy().astype(np.uint32))

        if not has_any_valid_move(board):
            break

    boards_u32 = np.stack(boards, axis=0)
    actions_u8 = np.array(actions, dtype=np.uint8)
    return boards_u32, actions_u8, int(score), int(np.max(board))


def improve_one_replay(path: str, cfg: SearchConfig, master_rng: random.Random) -> Tuple[SimulationResult, Dict]:
    boards, actions, meta = load_npz(path)
    base_stats = replay_stats(boards, actions, meta)
    cum_scores = cumulative_scores_from_replay(boards.astype(np.int64), actions.astype(np.int64))

    current_boards = boards.copy()
    current_actions = actions.copy()
    current_score = int(cum_scores[-1])
    current_max_tile = int(np.max(current_boards[-1]))

    best_result = SimulationResult(
        boards=current_boards.copy(),
        actions=current_actions.copy(),
        score=current_score,
        max_tile=current_max_tile,
        steps=int(len(current_actions)),
        heuristic_value=board_value(current_boards[-1].astype(np.int64)),
        rollback_used=0,
        pass_index=0,
        source_file=path,
    )

    trace = {
        "source_file": path,
        "base_score": base_stats["score"],
        "base_max_tile": base_stats["max_tile"],
        "passes": [],
    }

    for pass_idx in range(1, cfg.outer_passes + 1):
        pass_log = {
            "pass_index": pass_idx,
            "start_score": int(current_score),
            "start_max_tile": int(current_max_tile),
            "improved": False,
            "best_local_score": int(current_score),
            "best_local_max_tile": int(current_max_tile),
            "best_local_rollback": 0,
        }

        curr_cum = cumulative_scores_from_replay(current_boards.astype(np.int64), current_actions.astype(np.int64))
        local_best_boards = current_boards.copy()
        local_best_actions = current_actions.copy()
        local_best_score = int(current_score)
        local_best_max_tile = int(current_max_tile)
        local_best_steps = int(len(current_actions))
        local_best_h = board_value(current_boards[-1].astype(np.int64))
        local_best_rollback = 0

        max_rb = min(cfg.max_rollback, len(current_actions))
        for rollback in range(1, max_rb + 1):
            pivot_idx = len(current_boards) - 1 - rollback
            if pivot_idx < 0:
                continue

            prefix_boards = current_boards[: pivot_idx + 1].copy()
            prefix_actions = current_actions[:pivot_idx].copy()
            prefix_score = int(curr_cum[pivot_idx])
            start_board = current_boards[pivot_idx].astype(np.int64)

            for _ in range(cfg.attempts_per_pivot):
                sim_rng = random.Random(master_rng.randrange(1 << 62))
                tail_boards, tail_actions, final_score, final_max_tile = rollout_from_board(
                    start_board=start_board,
                    start_score=prefix_score,
                    rng=sim_rng,
                    cfg=cfg,
                )

                full_boards = np.concatenate([prefix_boards, tail_boards[1:]], axis=0)
                full_actions = np.concatenate([prefix_actions, tail_actions], axis=0)
                final_steps = int(len(full_actions))
                final_h = board_value(full_boards[-1].astype(np.int64))

                is_better = is_better_candidate(
                    final_score,
                    final_max_tile,
                    final_steps,
                    local_best_score,
                    local_best_max_tile,
                    local_best_steps,
                    cfg.target_tile,
                )
                if is_better or (
                    final_score == local_best_score
                    and final_max_tile == local_best_max_tile
                    and final_steps == local_best_steps
                    and final_h > local_best_h
                ):
                    local_best_boards = full_boards
                    local_best_actions = full_actions
                    local_best_score = int(final_score)
                    local_best_max_tile = int(final_max_tile)
                    local_best_steps = final_steps
                    local_best_h = final_h
                    local_best_rollback = rollback
                    pass_log["improved"] = True
                    pass_log["best_local_score"] = int(final_score)
                    pass_log["best_local_max_tile"] = int(final_max_tile)
                    pass_log["best_local_rollback"] = int(rollback)

#                    if cfg.verbose:
 #                       print(
  #                          f"[IMPROVE] {os.path.basename(path)} | pass={pass_idx} rollback={rollback} "
   #                         f"-> score={final_score} max_tile={final_max_tile} steps={final_steps}"
    #                    )

        trace["passes"].append(pass_log)

        if pass_log["improved"]:
            current_boards = local_best_boards
            current_actions = local_best_actions
            current_score = int(local_best_score)
            current_max_tile = int(local_best_max_tile)

            if is_better_candidate(
                current_score,
                current_max_tile,
                len(current_actions),
                best_result.score,
                best_result.max_tile,
                best_result.steps,
                cfg.target_tile,
            ):
                best_result = SimulationResult(
                    boards=current_boards.copy(),
                    actions=current_actions.copy(),
                    score=current_score,
                    max_tile=current_max_tile,
                    steps=int(len(current_actions)),
                    heuristic_value=local_best_h,
                    rollback_used=int(local_best_rollback),
                    pass_index=pass_idx,
                    source_file=path,
                )
        else:
            break

        if cfg.stop_after_first_target and current_max_tile >= cfg.target_tile:
            break

    return best_result, trace


# -------------------------------------------------
# Output pruning and file selection
# -------------------------------------------------

def prune_output_folder(folder: str, keep_k: int, target_tile: int):
    files = list_npz_files(folder)
    if len(files) <= keep_k:
        return

    rows = []
    for fp in files:
        try:
            meta = read_npz_meta_only(fp)
            score = int(meta.get("score", -1))
            mt = int(meta.get("max_tile", -1))
            ts = int(meta.get("saved_at_unix", 0))
            rows.append((fp, score, mt, ts))
        except Exception:
            continue

    rows.sort(key=lambda x: (1 if x[2] >= target_tile else 0, x[2], x[1], x[3]))
    for fp, _, _, _ in rows[: max(0, len(rows) - keep_k)]:
        try:
            os.remove(fp)
        except Exception:
            pass


def select_input_files(cfg: SearchConfig) -> List[str]:
    rows = []
    for fp in list_npz_files(cfg.input_dir):
        try:
            meta = read_npz_meta_only(fp)
            score = int(meta.get("score", -1))
            mt = int(meta.get("max_tile", -1))
        except Exception:
            continue

        if cfg.skip_if_already_target and mt >= cfg.target_tile:
            continue
        if cfg.min_source_max_tile is not None and mt < cfg.min_source_max_tile:
            continue
        if cfg.max_source_max_tile is not None and mt > cfg.max_source_max_tile:
            continue
        rows.append((fp, mt, score))

    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if cfg.max_files > 0:
        rows = rows[: cfg.max_files]
    return [fp for fp, _, _ in rows]


# -------------------------------------------------
# Main
# -------------------------------------------------

def run(cfg: SearchConfig):
    ensure_dir(cfg.output_dir)
    master_rng = random.Random(cfg.random_seed)
    source_files = select_input_files(cfg)

    if not source_files:
        print("No source NPZ files matched the current filters.")
        return

    print(f"[START] source_files={len(source_files)} input_dir={cfg.input_dir} output_dir={cfg.output_dir}")

    summary = {
        "config": asdict(cfg),
        "started_at_unix": int(time.time()),
        "processed": [],
    }

    global_best: Optional[Dict] = None

    for idx, fp in enumerate(source_files, start=1):
        print(f"\n[{idx}/{len(source_files)}] processing {os.path.basename(fp)}")
        base_boards, base_actions, base_meta = load_npz(fp)
        base = replay_stats(base_boards, base_actions, base_meta)

        improved, trace = improve_one_replay(fp, cfg, master_rng)
        improved_obj = objective_tuple(improved.score, improved.max_tile, improved.steps, cfg.target_tile)
        base_obj = objective_tuple(base["score"], base["max_tile"], base["steps"], cfg.target_tile)

        item = {
            "source_file": fp,
            "base_score": int(base["score"]),
            "base_max_tile": int(base["max_tile"]),
            "improved_score": int(improved.score),
            "improved_max_tile": int(improved.max_tile),
            "improved_steps": int(improved.steps),
            "rollback_used": int(improved.rollback_used),
            "pass_index": int(improved.pass_index),
            "saved_path": None,
            "trace": trace,
        }

        if improved_obj > base_obj:
            tag = (
                f"from_{sanitize_stem(fp)}_p{improved.pass_index:02d}_rb{improved.rollback_used:03d}"
            )
            saved = save_replay_npz(
                cfg.output_dir,
                file_tag=tag,
                score=int(improved.score),
                max_tile=int(improved.max_tile),
                steps=int(improved.steps),
                boards_u32=improved.boards.astype(np.uint32),
                actions_u8=improved.actions.astype(np.uint8),
                extra_meta={
                    "source_file": os.path.basename(fp),
                    "source_score": int(base["score"]),
                    "source_max_tile": int(base["max_tile"]),
                    "rollback_used": int(improved.rollback_used),
                    "pass_index": int(improved.pass_index),
                    "target_tile": int(cfg.target_tile),
                },
            )
            prune_output_folder(cfg.output_dir, cfg.keep_best_outputs, cfg.target_tile)
            item["saved_path"] = saved
            print(
                f"[SAVE] {os.path.basename(fp)} \n -> score {base['score']} -> {improved.score} ({100 * (improved.score - base['score']) / base['score']:.1f}% improved), "
                f"tile {base['max_tile']} -> {improved.max_tile} |\n {saved}"
            )
        else:
            print(
                f"[KEEP] no improvement for {os.path.basename(fp)} "
                f"(score {base['score']}, tile {base['max_tile']})"
            )

        if global_best is None or improved_obj > objective_tuple(
            global_best["score"], global_best["max_tile"], global_best["steps"], cfg.target_tile
        ):
            global_best = {
                "source_file": fp,
                "score": int(improved.score),
                "max_tile": int(improved.max_tile),
                "steps": int(improved.steps),
                "rollback_used": int(improved.rollback_used),
                "pass_index": int(improved.pass_index),
                "saved_path": item["saved_path"],
            }

        summary["processed"].append(item)

        if cfg.stop_after_first_target and improved.max_tile >= cfg.target_tile:
            print(f"[STOP] target tile {cfg.target_tile} reached. Stopping early.")
            break

    summary["finished_at_unix"] = int(time.time())
    summary["global_best"] = global_best

    ensure_dir(os.path.dirname(cfg.summary_json) or ".")
    with open(cfg.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[DONE]")
    if global_best is not None:
        print(
            f"Best found: score={global_best['score']} max_tile={global_best['max_tile']} "
            f"steps={global_best['steps']} source={os.path.basename(global_best['source_file'])}"
        )
    print(f"Summary saved to: {cfg.summary_json}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Load existing 2048 NPZ replays, roll back 1..n steps from terminal states, "
            "resimulate many alternative continuations, and save improved replays."
        )
    )
    p.add_argument("--input-dir", type=str, default="replays_best")
    p.add_argument("--output-dir", type=str, default="replays_improved")
    p.add_argument("--summary-json", type=str, default="replays_improved/summary.json")

    p.add_argument("--max-files", type=int, default=16)
    p.add_argument("--min-source-max-tile", type=int, default=256)
    p.add_argument("--max-source-max-tile", type=int, default=65536)
    p.add_argument("--include-already-target", action="store_true")

    p.add_argument("--target-tile", type=int, default=131072)
    p.add_argument("--max-rollback", type=int, default=128)
    p.add_argument("--attempts-per-pivot", type=int, default=16)
    p.add_argument("--outer-passes", type=int, default=8)
    p.add_argument("--max-steps-per-attempt", type=int, default=1500)

    p.add_argument("--preselect-top-actions", type=int, default=2)
    p.add_argument("--rollouts-per-action", type=int, default=0)
    p.add_argument("--rollout-depth", type=int, default=12)
    p.add_argument("--spawn-eval-samples", type=int, default=4)
    p.add_argument("--exploration-eps", type=float, default=0.06)
    p.add_argument("--top-rollout-bonus", type=float, default=0.20)

    p.add_argument("--keep-best-outputs", type=int, default=128)
    p.add_argument("--stop-after-first-target", action="store_true")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    return p


def args_to_config(args: argparse.Namespace) -> SearchConfig:
    skip_if_already_target = not args.include_already_target
    return SearchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        summary_json=args.summary_json,
        max_files=args.max_files,
        min_source_max_tile=args.min_source_max_tile,
        max_source_max_tile=args.max_source_max_tile,
        skip_if_already_target=skip_if_already_target,
        target_tile=args.target_tile,
        max_rollback=args.max_rollback,
        attempts_per_pivot=args.attempts_per_pivot,
        outer_passes=args.outer_passes,
        max_steps_per_attempt=args.max_steps_per_attempt,
        preselect_top_actions=args.preselect_top_actions,
        rollouts_per_action=args.rollouts_per_action,
        rollout_depth=args.rollout_depth,
        spawn_eval_samples=args.spawn_eval_samples,
        exploration_eps=args.exploration_eps,
        top_rollout_bonus=args.top_rollout_bonus,
        keep_best_outputs=args.keep_best_outputs,
        stop_after_first_target=args.stop_after_first_target,
        random_seed=args.random_seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    parser = build_argparser()
    config = args_to_config(parser.parse_args())
    run(config)
