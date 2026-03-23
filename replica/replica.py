import os
import re
import numpy as np


def transform_board(board: np.ndarray, idx: int) -> np.ndarray:
    """
    idx 1~7에 따라 4x4 보드를 변환
    0은 원본이라고 보고, 아래 7개만 생성:
      1: rot90 1회
      2: rot90 2회
      3: rot90 3회
      4: 좌우 대칭
      5: 좌우 대칭 + rot90 1회
      6: 좌우 대칭 + rot90 2회
      7: 좌우 대칭 + rot90 3회
    """
    if idx == 1:
        return np.rot90(board, 1)
    elif idx == 2:
        return np.rot90(board, 2)
    elif idx == 3:
        return np.rot90(board, 3)
    elif idx == 4:
        return np.fliplr(board)
    elif idx == 5:
        return np.rot90(np.fliplr(board), 1)
    elif idx == 6:
        return np.rot90(np.fliplr(board), 2)
    elif idx == 7:
        return np.rot90(np.fliplr(board), 3)
    else:
        raise ValueError(f"invalid transform idx: {idx}")


def transform_action(action: int, idx: int) -> int:
    """
    action: 0=up, 1=down, 2=left, 3=right
    보드 변환에 맞게 action도 같이 변환
    """
    # 방향 벡터로 표현
    action_to_vec = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }
    vec_to_action = {v: k for k, v in action_to_vec.items()}

    dr, dc = action_to_vec[int(action)]

    def rot90_vec(r, c):
        # np.rot90과 같은 방향(반시계 90도)
        return -c, r

    def fliplr_vec(r, c):
        # 좌우 대칭
        return r, -c

    if idx in (1, 2, 3):
        for _ in range(idx):
            dr, dc = rot90_vec(dr, dc)

    elif idx in (4, 5, 6, 7):
        dr, dc = fliplr_vec(dr, dc)
        for _ in range(idx - 4):
            dr, dc = rot90_vec(dr, dc)

    return vec_to_action[(dr, dc)]


def make_new_filename(filename: str, prefix_digit: int) -> str:
    """
    예:
    best_tile001024_score000009852_ep001454.npz
    ->
    best_tile001024_score000009852_ep101454.npz
    """
    m = re.match(r"^(.*_ep)(\d{6})(\.npz)$", filename)
    if not m:
        raise ValueError(f"파일명 형식이 예상과 다릅니다: {filename}")

    head, ep_num, ext = m.groups()
    new_ep = f"{prefix_digit}{ep_num[1:]}"  # 맨 앞자리만 1~7로 교체
    return f"{head}{new_ep}{ext}"


def duplicate_all_npz_in_same_folder():
    folder = os.path.dirname(os.path.abspath(__file__))
    npz_files = [f for f in os.listdir(folder) if f.endswith(".npz")]

    if not npz_files:
        print("같은 폴더에 npz 파일이 없습니다.")
        return

    for fname in sorted(npz_files):
        src_path = os.path.join(folder, fname)

        # 이미 생성된 ep1xxxxx ~ ep7xxxxx 파일은 다시 처리하지 않도록 선택적으로 제외
        m = re.match(r"^.*_ep(\d{6})\.npz$", fname)
        if not m:
            print(f"[SKIP] 이름 형식 불일치: {fname}")
            continue

        ep_num = m.group(1)
        if ep_num[0] in "1234567":
            print(f"[SKIP] 이미 생성된 파일로 보임: {fname}")
            continue

        try:
            data = np.load(src_path, allow_pickle=False)

            boards = data["boards"]       # shape: (T, 4, 4)
            actions = data["actions"]     # shape: (T-1,)
            meta = data["meta"]           # 그대로 저장

            for idx in range(1, 8):
                new_boards = np.array([transform_board(b, idx) for b in boards], dtype=boards.dtype)
                new_actions = np.array([transform_action(a, idx) for a in actions], dtype=actions.dtype)

                new_name = make_new_filename(fname, idx)
                new_path = os.path.join(folder, new_name)

                np.savez_compressed(
                    new_path,
                    boards=new_boards,
                    actions=new_actions,
                    meta=meta
                )

                print(f"[OK] {new_name}")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")


if __name__ == "__main__":
    duplicate_all_npz_in_same_folder()