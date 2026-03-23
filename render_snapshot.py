import os
import math
import json
import numpy as np

def try_import_pygame():
    try:
        import pygame
        return pygame
    except Exception:
        return None

def list_npz_files(folder: str):
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    files.sort()
    return [os.path.join(folder, f) for f in files]

def _lerp(a, b, t):
    return int(a + (b - a) * t)

def _lerp_color(c1, c2, t):
    return (_lerp(c1[0], c2[0], t), _lerp(c1[1], c2[1], t), _lerp(c1[2], c2[2], t))

def tile_color_general(v: int):
    if v == 0: return (205, 193, 180)
    p = math.log2(v)
    t = min(1.0, max(0.0, (p - 1.0) / 16.0))
    c_low = (238, 228, 218)
    c_high = (60, 58, 50)
    return _lerp_color(c_low, c_high, t)

def text_color(v: int):
    return (119, 110, 101) if v in (2, 4) else (249, 246, 242)

class ReplayViewer:
    def __init__(self, pygame, W=940, H=600):
        self.pg = pygame
        pygame.init()
        
        # --- 핵심 설정: 키 반복 입력 설정 ---
        # set_repeat(지연시간, 반복간격) 단위: 밀리초(ms)
        # 1000ms(1초) 동안 누르고 있으면, 그 뒤로는 50ms(0.05초)마다 키 입력 이벤트 발생
        self.pg.key.set_repeat(1000, 50) 
        
        self.W, self.H = W, H
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("2048 Replay Viewer")

        self.bg = (250, 248, 239)
        self.panel_bg = (187, 173, 160)
        self.pad = 16
        self.tile_gap = 10
        self.board_size = 440
        self.board_x = 40
        self.board_y = 110

        self.font_mid = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_tile_big = pygame.font.SysFont("consolas", 34, bold=True)
        self.font_tile_mid = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_tile_small = pygame.font.SysFont("consolas", 22, bold=True)

        self.running = True
        self.paused = True
        self.fps = 500
        self.clock = pygame.time.Clock()

    def _tile_font(self, v: int):
        digits = len(str(v))
        if digits <= 4: return self.font_tile_big
        if digits <= 5: return self.font_tile_mid
        return self.font_tile_small

    def draw_board(self, board: np.ndarray):
        pg = self.pg
        pg.draw.rect(self.screen, self.panel_bg, (self.board_x, self.board_y, self.board_size, self.board_size), border_radius=12)
        tile_w = (self.board_size - 2*self.pad - 3*self.tile_gap) // 4
        for r in range(4):
            for c in range(4):
                v = int(board[r, c])
                x = self.board_x + self.pad + c*(tile_w + self.tile_gap)
                y = self.board_y + self.pad + r*(tile_w + self.tile_gap)
                pg.draw.rect(self.screen, tile_color_general(v), (x, y, tile_w, tile_w), border_radius=10)
                if v != 0:
                    font = self._tile_font(v)
                    surf = font.render(str(v), True, text_color(v))
                    rect = surf.get_rect(center=(x + tile_w//2, y + tile_w//2))
                    self.screen.blit(surf, rect)

    def draw_header(self, meta: dict, idx: int, total: int, file_idx: int, file_total: int):
        color = (90, 80, 70)
        lines = [
            f"file: {file_idx+1}/{file_total}",
            f"episode: {meta.get('episode','?')}  score: {meta.get('score','?')}  max_tile: {meta.get('max_tile','?')}  steps: {meta.get('steps','?')}",
            f"frame: {idx+1}/{total}  fps: {self.fps}  paused: {self.paused}",
            "keys: SPACE pause | <-/-> frame | N/P file | Q/E 10 files | Up/Down fps | ESC quit"
        ]
        y = 10
        for s in lines:
            surf = self.font_mid.render(s, True, color)
            self.screen.blit(surf, (20, y))
            y += 22

    def handle_events(self):
        cmd = None
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT:
                self.running = False
            elif e.type == self.pg.KEYDOWN:
                # set_repeat 덕분에 1초 이상 누르면 KEYDOWN 이벤트가 반복해서 들어옵니다.
                if e.key == self.pg.K_ESCAPE: self.running = False
                elif e.key == self.pg.K_SPACE: self.paused = not self.paused
                elif e.key == self.pg.K_UP: self.fps = min(240, self.fps + 10)
                elif e.key == self.pg.K_DOWN: self.fps = max(1, self.fps - 10)
                elif e.key == self.pg.K_n: cmd = "next_file"
                elif e.key == self.pg.K_p: cmd = "prev_file"
                elif e.key == self.pg.K_e: cmd = "next_10_file"
                elif e.key == self.pg.K_q: cmd = "prev_10_file"
                elif e.key == self.pg.K_RIGHT: cmd = "next_frame"
                elif e.key == self.pg.K_LEFT: cmd = "prev_frame"
        return cmd

    def tick(self):
        self.clock.tick(self.fps)

def load_npz(path: str):
    data = np.load(path, allow_pickle=False)
    boards = data["boards"].astype(np.uint32)
    meta = json.loads(str(data["meta"]))
    return boards, meta

def main():
    replay_dir = "replays_improved"
    files = list_npz_files(replay_dir)
    if not files:
        print(f"No .npz replay files found in: {replay_dir}")
        return

    pygame = try_import_pygame()
    if pygame is None: return

    viewer = ReplayViewer(pygame)
    file_i = 0
    boards, meta = load_npz(files[file_i])
    idx = 0

    while viewer.running:
        cmd = viewer.handle_events()

        if cmd == "next_file":
            file_i = min(len(files)-1, file_i + 1)
            boards, meta = load_npz(files[file_i])
            idx = 0
        elif cmd == "prev_file":
            file_i = max(0, file_i - 1)
            boards, meta = load_npz(files[file_i])
            idx = 0
        elif cmd == "next_10_file":
            file_i = min(len(files)-1, file_i + 10)
            boards, meta = load_npz(files[file_i])
            idx = 0
        elif cmd == "prev_10_file":
            file_i = max(0, file_i - 10)
            boards, meta = load_npz(files[file_i])
            idx = 0
        elif cmd == "next_frame":
            idx = min(len(boards)-1, idx + 1)
        elif cmd == "prev_frame":
            idx = max(0, idx - 1)

        if not viewer.paused:
            if idx < len(boards) - 1:
                idx += 1
            else:
                viewer.paused = True

        viewer.screen.fill(viewer.bg)
        viewer.draw_header(meta, idx, len(boards), file_i, len(files))
        viewer.draw_board(boards[idx])
        pygame.display.flip()
        viewer.tick()

    pygame.quit()

if __name__ == "__main__":
    main()