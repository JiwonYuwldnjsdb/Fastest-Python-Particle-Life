import math, random
import numpy as np
import pygame, sys
from pygame.locals import *

from numba import njit, prange

pygame.init()

# ------------------------------
# 색상 팔레트
# ------------------------------
colorType = [
    (247, 52, 106),     # Red
    (19, 220, 213),     # Green
    (70, 48, 237),     # Blue
    (248, 209, 36),   # Yellow
    (203, 91, 204),   # Magenta
    (58, 216, 253),   # Cyan
    (249, 103, 51),   # Orange
    (120, 45, 255),   # Purple
    (8, 207, 165),   # Teal
    (255, 145, 203), # Pink
]

# 40x40 기본 타일(그리드에서 매 프레임 목표 크기로 스케일해 사용)
colorTypeSurface = []
for color in colorType:
    rect_surface = pygame.Surface((40,40), pygame.SRCALPHA)
    rect_surface.fill((*color,200))
    colorTypeSurface.append(rect_surface)

# ------------------------------
# 전체화면 윈도우 + 논리 캔버스
# ------------------------------
CANVAS_W, CANVAS_H = 1920, 1080  # 논리 캔버스(디자인 기준)
WINDOW = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)  # 실제 전체화면
pygame.display.set_caption("Particle Life Simulation")
SCREEN = pygame.Surface((CANVAS_W, CANVAS_H)).convert_alpha()

clock = pygame.time.Clock()

###### Load Song #####
pygame.mixer.init()
pygame.mixer.music.load('bgm.mp3')

ROBOTO = "Roboto-Light.ttf"

# ------------------------------
# 화면 스케일/표시 유틸
# ------------------------------
def _scale_info():
    ww, wh = WINDOW.get_size()
    s = min(ww / CANVAS_W, wh / CANVAS_H)     # aspect fit
    sw, sh = int(CANVAS_W * s), int(CANVAS_H * s)
    ox, oy = (ww - sw) // 2, (wh - sh) // 2   # 레터/필러 박스 오프셋
    return s, ox, oy, sw, sh

def present():
    s, ox, oy, sw, sh = _scale_info()
    WINDOW.fill((0, 0, 0))
    frame = pygame.transform.smoothscale(SCREEN, (sw, sh))
    WINDOW.blit(frame, (ox, oy))
    pygame.display.flip()

def window_to_logical(pos):
    """전체화면 좌표 -> 논리 캔버스 좌표. 바깥(레터박스) 클릭이면 None."""
    s, ox, oy, sw, sh = _scale_info()
    x, y = pos
    if x < ox or y < oy or x >= ox + sw or y >= oy + sh:
        return None
    lx = int((x - ox) / s)
    ly = int((y - oy) / s)
    return (lx, ly)

# ------------------------------
# UI 스케일 유틸 (논리 캔버스 기준)
# ------------------------------
BASE_W, BASE_H = 1920*2, 1080*2
UIS = min(CANVAS_W/BASE_W, CANVAS_H/BASE_H)

def S(x):   # 길이/좌표 스케일
    return int(round(x * UIS))

def FS(sz): # 폰트 크기 스케일 (가독성 최소값 12)
    return max(12, int(round(sz * UIS)))

# ------------------------------
# 그리드(상호작용 매트릭스) 표시
# ------------------------------
class AttractiveGrid:
    def __init__(self, grid=[], m=6):
        self.grid = grid
        self.m = m
        self.font = pygame.font.Font(ROBOTO, FS(15))
        self.cell = S(40)
        self.off  = S(10)

        self.rect_surface = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        self.rect_surface.fill((0,0,0,200))
        self.rect_color_cover = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
        self.rect_color_cover.fill((0,0,0,50))
    
    def render(self, SCREEN):
        global pend
        c = self.cell
        off = self.off

        # 좌상단 헤더(빈칸)
        SCREEN.blit(self.rect_surface, (off, off, c, c))
        pygame.draw.rect(SCREEN, (255,255,255), pygame.Rect(off, off, c, c), 1)
        
        # 상단 색상 라인
        for i in range(self.m):
            x = off + c * (i + 1)
            y = off
            color_surf = pygame.transform.smoothscale(colorTypeSurface[i], (c, c))
            SCREEN.blit(color_surf, (x, y))
            SCREEN.blit(self.rect_color_cover, (x, y, c, c))
        
        # 좌측 색상 라인
        for j in range(self.m):
            x = off
            y = off + c * (j + 1)
            color_surf = pygame.transform.smoothscale(colorTypeSurface[j], (c, c))
            SCREEN.blit(color_surf, (x, y))
            SCREEN.blit(self.rect_color_cover, (x, y, c, c))
        
        # 본 매트릭스
        for i in range(self.m):
            for j in range(self.m):
                x = off + c * (i + 1)
                y = off + c * (j + 1)
                SCREEN.blit(self.rect_surface, (x, y, c, c))
                pygame.draw.rect(SCREEN, (255,255,255), pygame.Rect(x, y, c, c), 1)
                
                if pend:
                    tmp_text = self.font.render("-", True, (255, 255, 255))
                    w,h = tmp_text.get_size()
                    SCREEN.blit(tmp_text, (x + (c - w)//2, y + (c - h)//2))
                else:
                    tmp_text = self.font.render(f"{self.grid[i][j]:.2f}", True, (255, 255, 255))
                    w,h = tmp_text.get_size()
                    SCREEN.blit(tmp_text, (x + c - w - S(3), y + (c - h)//2))

@njit(fastmath=True, inline='always')
def _torus_delta(d):
    if d > 0.5:
        return d - 1.0
    elif d < -0.5:
        return d + 1.0
    return d

@njit(fastmath=True, inline='always')
def _force_scalar(rn, a, beta):
    if rn < beta:
        return rn / beta - 1.0
    elif rn < 1.0:
        tri = 1.0 - abs(2.0 * rn - 1.0 - beta) / (1.0 - beta)
        return a * tri
    else:
        return 0.0

@njit(parallel=True, fastmath=True)
def step_kernel(posx, posy, velx, vely, col, A,
                cell_x, cell_y, starts, order,
                nx, ny, rMax, dt, frictionFactor, forceFactor):
    n = posx.shape[0]
    beta = 0.3
    for i in prange(n):
        cxi = cell_x[i]
        cyi = cell_y[i]
        tfx = 0.0
        tfy = 0.0
        # 3x3 이웃
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                nxn = cxi + ddx
                if nxn < 0:
                    nxn += nx
                elif nxn >= nx:
                    nxn -= nx
                nyn = cyi + ddy
                if nyn < 0:
                    nyn += ny
                elif nyn >= ny:
                    nyn -= ny
                c2 = nxn + nx * nyn
                j0 = starts[c2]
                j1 = starts[c2 + 1]
                for k in range(j0, j1):
                    j = order[k]
                    if j == i:
                        continue
                    rx = posx[j] - posx[i]
                    ry = posy[j] - posy[i]
                    rx = _torus_delta(rx)
                    ry = _torus_delta(ry)
                    r = math.hypot(rx, ry)
                    if (r > 0.0) and (r < rMax):
                        rn = r / rMax
                        a = A[col[i], col[j]]
                        f = _force_scalar(rn, a, beta)
                        invr = 1.0 / r
                        tfx += (rx * invr) * f
                        tfy += (ry * invr) * f
        tfx *= rMax * forceFactor
        tfy *= rMax * forceFactor
        vx = velx[i] * frictionFactor + tfx * dt
        vy = vely[i] * frictionFactor + tfy * dt
        px = posx[i] + vx * dt
        py = posy[i] + vy * dt
        px = px - math.floor(px)  # mod 1.0
        py = py - math.floor(py)
        velx[i] = vx
        vely[i] = vy
        posx[i] = px
        posy[i] = py

@njit(parallel=True, fastmath=True)
def blast_kernel(posx, posy, velx, vely, cx, cy, radius, strength):
    n = posx.shape[0]
    for i in prange(n):
        rx = posx[i] - cx
        ry = posy[i] - cy
        rx = _torus_delta(rx)
        ry = _torus_delta(ry)
        r = math.hypot(rx, ry)
        if (r > 0.0) and (r < radius):
            s = (1.0 - r / radius) * strength
            invr = 1.0 / r
            velx[i] += (rx * invr) * s
            vely[i] += (ry * invr) * s

class Simulation:
    def __init__(self, n = 1000, dt = 0.02, frictionHalfLife = 0.04, rMax = 0.1, m = 6, forceFactor = 10):
        self.m = int(m)
        self.dt = float(dt)
        self.frictionHalfLife = float(frictionHalfLife)
        self.rMax = float(rMax)
        self.forceFactor = float(forceFactor)

        # 상태 배열 (분리 보관: numba 커널에서 1D가 빠름)
        self.posx = np.random.rand(int(n)).astype(np.float32)
        self.posy = np.random.rand(int(n)).astype(np.float32)
        self.velx = np.zeros(int(n), dtype=np.float32)
        self.vely = np.zeros(int(n), dtype=np.float32)
        self.col  = np.random.randint(0, self.m, size=int(n), dtype=np.int32)

        # 상호작용 행렬
        self.A = np.random.uniform(-1, 1, (self.m, self.m)).astype(np.float32)

        self.frictionFactor = np.float32(0.5 ** (self.dt / self.frictionHalfLife))

        # ---- 기존 코드 호환용 별칭 ----
        self._rebind_views()

    def _rebind_views(self):
        # grid에서 쓰는 이름
        self.matrix = self.A
        # 렌더에서 쓰는 이름 (numpy 1D view 흉내)
        self.positionsX = self.posx
        self.positionsY = self.posy
        self.velocitiesX = self.velx
        self.velocitiesY = self.vely
        self.colors = self.col

    def _resize(self, new_n: int):
        old_n = self.posx.shape[0]
        new_n = int(new_n)
        if new_n == old_n:
            return
        if new_n < old_n:
            self.posx = self.posx[:new_n].copy()
            self.posy = self.posy[:new_n].copy()
            self.velx = self.velx[:new_n].copy()
            self.vely = self.vely[:new_n].copy()
            self.col  = self.col[:new_n].copy()
        else:
            extra = new_n - old_n
            self.posx = np.concatenate([self.posx, np.random.rand(extra).astype(np.float32)])
            self.posy = np.concatenate([self.posy, np.random.rand(extra).astype(np.float32)])
            self.velx = np.concatenate([self.velx, np.zeros(extra, dtype=np.float32)])
            self.vely = np.concatenate([self.vely, np.zeros(extra, dtype=np.float32)])
            self.col  = np.concatenate([self.col , np.random.randint(0, self.m, size=extra, dtype=np.int32)])
        self._rebind_views()

    @property
    def n(self) -> int:
        return int(self.posx.shape[0])

    @n.setter
    def n(self, new_n: int):
        self._resize(int(new_n))

    def updateParticles(self):
        # --- 셀 구축 (NumPy) ---
        cell_size = self.rMax
        nx = int(1.0 / cell_size)
        ny = int(1.0 / cell_size)
        total_cells = nx * ny

        cell_x = np.floor(self.posx / cell_size).astype(np.int32) % nx
        cell_y = np.floor(self.posy / cell_size).astype(np.int32) % ny
        cid = cell_x + nx * cell_y
        order = np.argsort(cid).astype(np.int32)
        cid_sorted = cid[order]
        counts = np.bincount(cid_sorted, minlength=total_cells).astype(np.int32)
        starts = np.empty(total_cells + 1, dtype=np.int32)
        starts[0] = 0
        np.cumsum(counts, out=starts[1:])

        # --- Numba 커널 실행 (in-place 업데이트) ---
        step_kernel(self.posx, self.posy, self.velx, self.vely, self.col, self.A,
                    cell_x, cell_y, starts, order,
                    nx, ny, self.rMax, self.dt, self.frictionFactor, self.forceFactor)

        self._rebind_views()

    def renderScreen(self, SCREEN, pend = False):
        SCREEN.fill((0, 0, 0))
        if not pend:
            r = max(1, S(2))
            X = (self.posx * CANVAS_W).astype(np.int32)
            Y = (self.posy * CANVAS_H).astype(np.int32)
            col = self.col
            for x, y, c in zip(X, Y, col):
                pygame.draw.circle(SCREEN, colorType[int(c)], (int(x), int(y)), r)

    def blast(self, cx, cy, radius=None, strength=None):
        if radius is None:
            radius = self.rMax * 1.5
        if strength is None:
            strength = 3.5
        blast_kernel(self.posx, self.posy, self.velx, self.vely,
                     np.float32(cx), np.float32(cy),
                     np.float32(radius), np.float32(strength))
        self._rebind_views()

# ------------------------------
# 백엔드 선택
# ------------------------------

# ------------------------------
# UI 위젯
# ------------------------------
class Dropdown:
    def __init__(self, x, y, width = 330, height = 35, options = [], font = None, default_text="Select an option", id = None):
        if font is None:
            font = pygame.font.Font(ROBOTO, FS(21))
        self.rect = pygame.Rect(x, y, width, height)
        self.rectInfo = (x,y,width,height)
        self.bgcolor = (255,255,255)
        self.tgcolor = (255,255,255)
        self.id = id
        self.tg_surface = pygame.Surface((width,height), pygame.SRCALPHA)
        self.tg_surface.fill((0,0,0,200))
        self.slcolor = (0,0,0)
        self.h_surface = pygame.Surface((width,height), pygame.SRCALPHA)
        self.h_surface.fill((255,255,255,30))
        self.options = options
        self.font = font
        self.selected = default_text
        self.expanded = False
        self.hovered_option = -1
        self.rect_surface = pygame.Surface((width,height), pygame.SRCALPHA)
        self.rect_surface.fill((0,0,0,200))

    def draw(self, surface):
        surface.blit(self.rect_surface, self.rectInfo)
        pygame.draw.rect(surface, self.bgcolor, self.rect, 1)
        text = self.font.render(self.selected, True, self.tgcolor)
        surface.blit(text, (self.rect.x + S(10), self.rect.y + S(5)))

        if self.expanded:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                if i != self.hovered_option:
                    surface.blit(self.tg_surface, option_rect)
                else:
                    surface.blit(self.h_surface, option_rect)
                    pygame.draw.rect(surface, self.bgcolor, option_rect, 1)
                
                option_text = self.font.render(option, True, self.tgcolor)
                surface.blit(option_text, (option_rect.x + S(10), option_rect.y + S(5)))

    def handle_event(self, event):
        global pend
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
            elif self.expanded:
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                    if option_rect.collidepoint(event.pos):
                        if self.id != 'm' or pend:
                            self.selected = option
                        self.expanded = False
                        return option
        elif event.type == pygame.MOUSEMOTION:
            if self.expanded:
                self.hovered_option = -1
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                    if option_rect.collidepoint(event.pos):
                        self.hovered_option = i
                        break

class GameObject:
    def __init__(self, x, y, width, height, font, screen):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = font
        self.screen = screen

class DialogBox(GameObject):
    def __init__(self, screen, title, text, width=400, height=200):
        self.title = title
        self.text = text
        self.title_font = pygame.font.Font(ROBOTO, FS(24))
        self.text_lines = text.split('\n')

        super().__init__((CANVAS_W - width) // 2,
                         (CANVAS_H - height) // 2,
                         width,
                         height,
                         pygame.font.Font(ROBOTO, FS(24)),
                         screen)
        
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.border_color = (255,255,255)
        self.text_color = (255,255,255)
        self.dialog_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.dialog_surface.fill((0,0,0,200))

    def draw(self):
        self.screen.blit(self.dialog_surface, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(self.screen, self.border_color, self.rect, 1)

        title_surface = self.title_font.render(self.title, True, self.text_color)
        self.screen.blit(title_surface, (self.x + S(20), self.y + S(10)))

        pygame.draw.line(self.screen, self.border_color,
                         (self.x + S(20), self.y + S(50)),
                         (self.x + self.width - S(20), self.y + S(50)), 1)

        for i, line in enumerate(self.text_lines):
            line_surface = self.font.render(line, True, self.text_color)
            self.screen.blit(line_surface, (self.x + S(20), self.y + S(60) + i * S(30)))

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return True
            elif event.key == pygame.K_ESCAPE:
                return False

class Button(GameObject):
    def __init__(self, x, y, width = 330, height = 35,  text = "Restart", screen = None):
        self.text = text
        self.clicked = False
        self.rect_surface = pygame.Surface((width,height), pygame.SRCALPHA)
        self.rect_surface.fill((0,0,0,200))
        super().__init__(x,y,width,height,pygame.font.Font(ROBOTO, FS(21)),screen)
    
    def draw(self):
        self.screen.blit(self.rect_surface, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(self.screen, (255,255,255), (self.x, self.y, self.width, self.height), 1)
        text_surface = self.font.render(self.text, True, (255,255,255))
        text_rect = text_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        self.screen.blit(text_surface, text_rect)
    
    def is_hovered(self, mouse_pos):
        return self.x <= mouse_pos[0] <= self.x + self.width and self.y <= mouse_pos[1] <= self.y + self.height

    def handle_event(self, event, settings, mouse_pos=None):
        # 논리 좌표 주입(없으면 기존 방식)
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        if self.is_hovered(mouse_pos):
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.rect_surface.fill((0,0,0,0)); self.rect_surface.fill((255,255,255,50))
                self.clicked = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.clicked:
                    self.rect_surface.fill((0,0,0,0)); self.rect_surface.fill((255,255,255,30))
                    self.clicked = False
                    return Simulation(m = settings[0], n = settings[1], forceFactor=settings[2])
            else:
                self.rect_surface.fill((0,0,0,0)); self.rect_surface.fill((255,255,255,30))
        else:
            self.rect_surface.fill((0,0,0,0)); self.rect_surface.fill((0,0,0,200))

# ------------------------------
# 메뉴 화면
# ------------------------------
def menuScreen():
    running = True
    currSimul = Simulation(m=random.randint(2,3))
    
    titleH1 = pygame.font.Font(ROBOTO, FS(72))
    text_titleH1 = titleH1.render("Particle Life", True, (255, 255, 255))
    wH1,hH1 = text_titleH1.get_size()
    
    titleP = pygame.font.Font(ROBOTO, FS(36))
    text_titleP = titleP.render("Space to Start", True, (255, 255, 255))
    wP,hP = text_titleP.get_size()
    
    x_positions = [(CANVAS_W - wH1) // 2, (CANVAS_W - wP) // 2]
    y_positions = [(CANVAS_H - hH1) // 7*3, (CANVAS_H - hP) // 7*4]
    
    rect_surface = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
    rect_surface.fill((0,0,0,100))

    while running:
        currSimul.updateParticles()
        currSimul.renderScreen(SCREEN)
        SCREEN.blit(rect_surface, (0,0,CANVAS_W,CANVAS_H))
        
        SCREEN.blit(text_titleH1, (x_positions[0], y_positions[0]))
        SCREEN.blit(text_titleP, (x_positions[1], y_positions[1]))
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return 2

        clock.tick(60)
        present()   # 전체화면 윈도우에 비율맞춰 표시
    
    return 0

# ------------------------------
# 본 시뮬레이션 화면
# ------------------------------
def runSimulation():
    global pend
    running = True
    
    tmpM = None
    currSimul = None
    currGrid = None
    gridTrigger = True
    spacePressed = False

    fontP = pygame.font.Font(ROBOTO, FS(21))

    dropdownPs = []
    activeDropDown = None
    nonActiveDropDown = []
    
    # 오른쪽 패널 기준 위치(논리 캔버스 기준)
    PANEL_W = S(360)  # 드롭다운 330 + 여유 30
    panel_x = CANVAS_W - PANEL_W

    # 라벨들
    m_text = fontP.render("Colors", True, (255, 255, 255)); dropdownPs.append([m_text, *m_text.get_size()])
    n_text = fontP.render("Particles", True, (255, 255, 255)); dropdownPs.append([n_text, *n_text.get_size()])
    f_text = fontP.render("Force", True, (255, 255, 255)); dropdownPs.append([f_text, *f_text.get_size()])
    
    # 드롭다운(스케일된 위치/크기)
    m_dropdown = Dropdown(panel_x, S(35),  width=S(330), height=S(35), options=[str(i) for i in range(1,11)], default_text="6", id='m')
    n_dropdown = Dropdown(panel_x, S(115), width=S(330), height=S(35), options=[str(i) for i in range(1000, 10001, 1000)], default_text="1000", id='n')
    f_dropdown = Dropdown(panel_x, S(195), width=S(330), height=S(35), options=[str(i) for i in range(1,11)], default_text="10", id='f')
    nonActiveDropDown += [m_dropdown, n_dropdown, f_dropdown]
    
    start_btn = Button(panel_x, S(255), width=S(330), height=S(35), screen=SCREEN)

    currDialog = None
    pend = False
    
    settings = [6,1000,10]
    
    while running:
        if currSimul:
            if not gridTrigger and not pend:
                currSimul.updateParticles()
            currSimul.renderScreen(SCREEN, pend)
        else:
            SCREEN.fill((0,0,0))
        
        if gridTrigger:
            if currGrid:
                currGrid.render(SCREEN)
            
            start_btn.draw()
            
            # 라벨: 드롭다운 왼쪽 정렬 + 바로 위
            LABEL_GAP = S(28)
            SCREEN.blit(dropdownPs[0][0], (m_dropdown.rect.x, max(S(10), m_dropdown.rect.y - LABEL_GAP)))
            SCREEN.blit(dropdownPs[1][0], (n_dropdown.rect.x, max(S(10), n_dropdown.rect.y - LABEL_GAP)))
            SCREEN.blit(dropdownPs[2][0], (f_dropdown.rect.x, max(S(10), f_dropdown.rect.y - LABEL_GAP)))

            for dd in nonActiveDropDown:
                dd.draw(SCREEN)
            if activeDropDown:
                activeDropDown.draw(SCREEN)

        if currDialog:
            currDialog.draw()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0

            # ---- 마우스 좌표를 논리 좌표로 변환 ----
            logical_pos = None
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                logical_pos = window_to_logical(event.pos) if hasattr(event, "pos") else None

            if currDialog:
                tmpDia = currDialog.handle_event(event)
                if tmpDia is not None:
                    if tmpDia == True:
                        pend = True
                        m_dropdown.selected = str(tmpM)
                        settings[0] = tmpM
                    currDialog = None
            else:
                if gridTrigger:
                    if activeDropDown:
                        # 드롭다운에 논리 좌표 전달
                        ev = event
                        if logical_pos is not None:
                            ev = pygame.event.Event(event.type, {'pos': logical_pos, 'button': getattr(event, 'button', 0)})
                        if tmp := activeDropDown.handle_event(ev):
                            if activeDropDown.id == 'm':
                                tmpM = int(tmp)
                                if not pend:
                                    currDialog = DialogBox(SCREEN, "Notice", "You have to re-run the simulation\nfor this action.")
                                else:
                                    settings[0] = tmpM
                            elif activeDropDown.id == 'n':
                                settings[1] = int(tmp)
                                if currSimul:
                                    currSimul.n = int(tmp)  # 배열 리사이즈까지 지원
                            elif activeDropDown.id == 'f':
                                settings[2] = int(tmp)
                                if currSimul:
                                    currSimul.forceFactor = float(tmp)
                        
                        if activeDropDown.expanded == False:
                            nonActiveDropDown.append(activeDropDown)
                            activeDropDown = None
                    else:
                        # 비활성 드롭다운들 순서대로 열기 시도 (각각 논리 좌표 전달)
                        if not activeDropDown:
                            ev = event
                            if logical_pos is not None:
                                ev = pygame.event.Event(event.type, {'pos': logical_pos, 'button': getattr(event, 'button', 0)})
                            nonActiveDropDown[0].handle_event(ev)
                            if nonActiveDropDown[0].expanded:
                                activeDropDown = nonActiveDropDown[0]
                                nonActiveDropDown.remove(activeDropDown)
                        
                        if not activeDropDown:
                            ev = event
                            if logical_pos is not None:
                                ev = pygame.event.Event(event.type, {'pos': logical_pos, 'button': getattr(event, 'button', 0)})
                            nonActiveDropDown[1].handle_event(ev)
                            if nonActiveDropDown[1].expanded:
                                activeDropDown = nonActiveDropDown[1]
                                nonActiveDropDown.remove(activeDropDown)
                        
                        if not activeDropDown:
                            ev = event
                            if logical_pos is not None:
                                ev = pygame.event.Event(event.type, {'pos': logical_pos, 'button': getattr(event, 'button', 0)})
                            nonActiveDropDown[2].handle_event(ev)
                            if nonActiveDropDown[2].expanded:
                                activeDropDown = nonActiveDropDown[2]
                                nonActiveDropDown.remove(activeDropDown)
                        
                        if not activeDropDown:
                            # 시작 버튼에도 논리 좌표 전달
                            tmp = start_btn.handle_event(event, settings, mouse_pos=(logical_pos if logical_pos else (-1,-1)))
                            if tmp:
                                currSimul = tmp
                                currGrid = AttractiveGrid(grid=currSimul.matrix, m=settings[0])
                                pend = False
                                gridTrigger = False
                
                # === 실행 중(그리드 off & pend 아님) 마우스 클릭 폭발 ===
                if not gridTrigger and not pend and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if logical_pos:
                        mx, my = logical_pos
                        cx = mx / CANVAS_W
                        cy = my / CANVAS_H
                        currSimul.blast(cx, cy)

                # 키 입력
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        spacePressed = True
                    if event.key == pygame.K_ESCAPE:
                        return 1
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        if spacePressed:
                            spacePressed = False
                            if gridTrigger:
                                gridTrigger = False
                                if activeDropDown:
                                    activeDropDown.expanded = False
                                    nonActiveDropDown.append(activeDropDown)
                                    activeDropDown = None
                            else:
                                gridTrigger = True
        
        clock.tick(60)
        present()   # 반드시 present로 표시
    
    return 1

# ------------------------------
# 엔트리 포인트
# ------------------------------
def main():
    currScreen = 1
    while True:
        if currScreen == 0:
            break
        elif currScreen == 1:
            currScreen = menuScreen()
        elif currScreen == 2:
            currScreen = runSimulation()

if __name__ == "__main__":
    pygame.mixer.music.play(-1)
    main()
    pygame.mixer.music.stop()
