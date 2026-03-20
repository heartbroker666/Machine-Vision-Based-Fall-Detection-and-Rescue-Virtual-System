"""
visualize_grid.py
场景栅格地图可视化 + A* 路径规划结果叠加
直接运行即可生成 grid_map_astar.png（论文图4-6）
依赖：pip install matplotlib numpy
"""

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════
#  地图参数（与 pr2_rescue.py 完全一致）
# ══════════════════════════════════════════════
GRID_RES    = 0.1    # 栅格分辨率（米/格）
ROBOT_CLEAR = 0.35   # 障碍膨胀半径（米）
MAP_X_MIN, MAP_X_MAX = -9.9,  0.0
MAP_Y_MIN, MAP_Y_MAX = -6.6,  0.0

# 起点（PR2初始位置）和终点（跌倒位置）
START_W = (-0.95, -1.48)
GOAL_W  = (-3.47, -6.43)

OBSTACLES = [
    # 外围墙壁
    ( 0.00, -0.65, 0.5),
    ( 0.00, -4.30, 0.5),
    (-9.90, -2.30, 0.5),
    (-3.80,  0.00, 0.6),
    # wall9 隔断墙（沿Y轴每0.3m一个点）
    (-3.30, -0.15, 0.5),
    (-3.30, -0.45, 0.5),
    (-3.30, -0.75, 0.5),
    (-3.30, -1.05, 0.5),
    (-3.30, -1.35, 0.5),
    (-3.30, -1.65, 0.5),
    (-3.30, -1.80, 0.5),
    (-3.30, -2.10, 0.5),
    (-3.30, -2.40, 0.5),
    (-3.30, -2.70, 0.5),
    (-3.30, -3.00, 0.5),
    (-3.30, -3.30, 0.5),
    (-3.30, -3.45, 0.5),
    # 餐桌椅
    (-1.07, -4.94, 0.7),
    (-1.46, -5.41, 0.4),
    (-0.64, -4.44, 0.4),
    (-1.39, -4.49, 0.4),
    (-0.71, -5.38, 0.4),
    # 书桌椅
    (-5.13, -0.51, 0.6),
    (-5.15, -0.89, 0.4),
    # 沙发区
    (-7.09, -2.55, 0.6),
    (-7.09, -2.06, 0.6),
    (-5.45, -2.51, 0.7),
    (-7.06, -0.80, 0.3),
    (-9.23, -2.92, 0.5),
    # 厨房
    (-0.52, -0.50, 0.4),
    (-1.31, -0.15, 0.4),
    (-2.19, -0.15, 0.4),
    (-2.85, -0.58, 0.4),
    # 书柜植物
    (-3.47, -6.43, 0.4),
    (-4.52, -6.08, 0.4),
]


# ══════════════════════════════════════════════
#  坐标转换
# ══════════════════════════════════════════════
def w2g(wx, wy):
    """世界坐标 → 栅格坐标"""
    return int((wx - MAP_X_MIN) / GRID_RES), int((wy - MAP_Y_MIN) / GRID_RES)

def g2w(gx, gy):
    """栅格坐标 → 世界坐标（格子中心）"""
    return gx * GRID_RES + MAP_X_MIN + GRID_RES / 2, \
           gy * GRID_RES + MAP_Y_MIN + GRID_RES / 2


# ══════════════════════════════════════════════
#  构建栅格地图
# ══════════════════════════════════════════════
def build_grid():
    cols = int((MAP_X_MAX - MAP_X_MIN) / GRID_RES) + 2
    rows = int((MAP_Y_MAX - MAP_Y_MIN) / GRID_RES) + 2
    grid = np.zeros((rows, cols), dtype=np.uint8)
    for (ox, oy, r) in OBSTACLES:
        inflate = r + ROBOT_CLEAR
        for gy in range(rows):
            for gx in range(cols):
                wx, wy = g2w(gx, gy)
                if math.sqrt((wx - ox)**2 + (wy - oy)**2) <= inflate:
                    grid[gy][gx] = 1
    return grid, rows, cols


# ══════════════════════════════════════════════
#  A* 路径规划
# ══════════════════════════════════════════════
def nearest_free(grid, rows, cols, gx, gy):
    """若格子在障碍内，向外搜索最近空闲格"""
    if 0 <= gx < cols and 0 <= gy < rows and grid[gy][gx] == 0:
        return gx, gy
    for r in range(1, 8):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                nx, ny = gx+dx, gy+dy
                if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
                    return nx, ny
    return gx, gy

def astar(grid, rows, cols, start_w, goal_w):
    sx, sy = w2g(*start_w)
    gx, gy = w2g(*goal_w)
    sx, sy = max(0,min(cols-1,sx)), max(0,min(rows-1,sy))
    gx, gy = max(0,min(cols-1,gx)), max(0,min(rows-1,gy))
    sx, sy = nearest_free(grid, rows, cols, sx, sy)
    gx, gy = nearest_free(grid, rows, cols, gx, gy)

    def h(x, y): return math.sqrt((x-gx)**2 + (y-gy)**2)

    open_set  = [(h(sx,sy), 0.0, sx, sy)]
    came_from = {}
    g_score   = {(sx,sy): 0.0}
    DIRS = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
            (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]

    while open_set:
        _, cost, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in came_from and g_score.get((cx,cy),1e9) < cost:
            continue
        if cx == gx and cy == gy:
            path = []
            node = (cx, cy)
            while node in came_from:
                path.append(g2w(*node))
                node = came_from[node]
            path.reverse()
            path.append(goal_w)
            return path
        for dx, dy, step in DIRS:
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < cols and 0 <= ny < rows): continue
            if grid[ny][nx] == 1: continue
            ng = g_score.get((cx,cy), 1e9) + step
            if ng < g_score.get((nx,ny), 1e9):
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+h(nx,ny), ng, nx, ny))
    return None

def simplify(path, grid, rows, cols):
    """路径简化：能直线到达就跳过中间点"""
    if len(path) <= 2:
        return path
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            steps = max(int(math.sqrt((path[j][0]-path[i][0])**2 +
                                      (path[j][1]-path[i][1])**2) / GRID_RES * 3), 3)
            clear = True
            for k in range(steps + 1):
                t  = k / steps
                wx = path[i][0] + t * (path[j][0] - path[i][0])
                wy = path[i][1] + t * (path[j][1] - path[i][1])
                gx_, gy_ = w2g(wx, wy)
                gx_ = max(0, min(cols-1, gx_))
                gy_ = max(0, min(rows-1, gy_))
                if grid[gy_][gx_] == 1:
                    clear = False
                    break
            if clear:
                break
            j -= 1
        result.append(path[j])
        i = j
    return result


# ══════════════════════════════════════════════
#  绘图主函数
# ══════════════════════════════════════════════
def draw(save_path="grid_map_astar.png"):
    grid, rows, cols = build_grid()

    # 路径规划
    raw_path = astar(grid, rows, cols, START_W, GOAL_W)
    if raw_path:
        path     = simplify(raw_path, grid, rows, cols)
        path_len = sum(
            math.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2)
            for i in range(len(path)-1))
        print(f"[A*] 原始路径: {len(raw_path)} 点  "
              f"简化后: {len(path)} 点  总长: {path_len:.2f} m")
    else:
        print("[A*] 未找到路径")
        path     = []
        path_len = 0

    # ── 画布 ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=150)

    # 底图：障碍=深灰，可通行=白
    display = np.ones((rows, cols, 3))
    display[grid == 1] = [0.22, 0.22, 0.25]
    ax.imshow(display, origin="lower",
              extent=[MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX],
              aspect="equal", interpolation="nearest")

    # 原始路径（浅蓝虚线）
    if raw_path:
        ax.plot([p[0] for p in raw_path],
                [p[1] for p in raw_path],
                '--', color='#aaccff', linewidth=0.8,
                alpha=0.45, zorder=3, label='原始 A* 路径')

    # 简化路径（青绿实线 + 路径点）
    if path:
        ax.plot([p[0] for p in path],
                [p[1] for p in path],
                '-o', color='#00d4aa', linewidth=2.2,
                markersize=5, markerfacecolor='white',
                markeredgecolor='#00d4aa', markeredgewidth=1.5,
                zorder=5, label=f'简化路径（{len(path)} 个路径点）')

        # 路径点编号（跳过起终点）
        for i, (wpx, wpy) in enumerate(path):
            if 0 < i < len(path) - 1:
                ax.annotate(str(i), xy=(wpx, wpy),
                            xytext=(wpx+0.12, wpy+0.15),
                            fontsize=6.5, color='#00ffcc', zorder=6)

        # 路径长度标注
        mid = path[len(path) // 2]
        ax.text(mid[0]+0.15, mid[1]-0.55,
                f'路径总长：{path_len:.2f} m',
                fontsize=8.5, color='#00d4aa',
                fontfamily='SimHei',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2333',
                          edgecolor='#00d4aa', alpha=0.88),
                zorder=9)

    # 起点
    ax.plot(START_W[0], START_W[1], 's', color='#4f8ef7',
            markersize=12, markeredgecolor='white',
            markeredgewidth=1.5, zorder=7)
    ax.annotate(f'起点 PR2\n({START_W[0]}, {START_W[1]})',
                xy=START_W, xytext=(START_W[0]+0.5, START_W[1]+0.65),
                fontsize=8, color='#4f8ef7', fontfamily='SimHei',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#4f8ef7', lw=1.0),
                zorder=8)

    # 终点
    ax.plot(GOAL_W[0], GOAL_W[1], '*', color='#ff3b5c',
            markersize=16, markeredgecolor='white',
            markeredgewidth=1.2, zorder=7)
    ax.annotate(f'终点（跌倒位置）\n({GOAL_W[0]}, {GOAL_W[1]})',
                xy=GOAL_W, xytext=(GOAL_W[0]-2.4, GOAL_W[1]-0.75),
                fontsize=8, color='#ff3b5c', fontfamily='SimHei',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ff3b5c', lw=1.0),
                zorder=8)

    # 隔断墙标注
    ax.annotate('隔断墙 wall9',
                xy=(-3.3, -1.8), xytext=(-1.9, -0.85),
                fontsize=8, color='#ffaa00', fontfamily='SimHei',
                arrowprops=dict(arrowstyle='->', color='#ffaa00', lw=0.9))

    # ── 坐标轴 ────────────────────────────────
    ax.set_xlabel('X 轴（m）', fontsize=10, fontfamily='SimHei')
    ax.set_ylabel('Y 轴（m）', fontsize=10, fontfamily='SimHei')
    ax.set_title('A* 全局路径规划结果（栅格分辨率 0.1 m，机器人安全间距 0.35 m）',
                 fontsize=11, fontfamily='SimHei', pad=10)
    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MIN, MAP_Y_MAX)
    ax.set_xticks(np.arange(MAP_X_MIN, MAP_X_MAX + 1, 1.0))
    ax.set_yticks(np.arange(MAP_Y_MIN, MAP_Y_MAX + 1, 1.0))
    ax.tick_params(labelsize=8)
    ax.grid(color='#555555', linewidth=0.25, alpha=0.4)

    # ── 图例 ──────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor='#383840', edgecolor='#888',
                       label='障碍物（含膨胀区域）'),
        mpatches.Patch(facecolor='white',   edgecolor='#aaa',
                       label='可通行区域'),
        plt.Line2D([0],[0], color='#aaccff', linewidth=1.2,
                   linestyle='--', label='原始 A* 路径'),
        plt.Line2D([0],[0], color='#00d4aa', linewidth=2,
                   marker='o', markersize=5, markerfacecolor='white',
                   label='简化路径（路径点）'),
        plt.Line2D([0],[0], marker='s', color='w',
                   markerfacecolor='#4f8ef7', markersize=9,
                   label='PR2 起点'),
        plt.Line2D([0],[0], marker='*', color='w',
                   markerfacecolor='#ff3b5c', markersize=11,
                   label='跌倒目标'),
    ]
    ax.legend(handles=legend_items, loc='lower left',
              prop={'family': 'SimHei', 'size': 8},
              framealpha=0.88, facecolor='#1a1a2e',
              edgecolor='#444', labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[完成] 图片已保存至 {save_path}")
    plt.show()


# ══════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    draw("grid_map_astar.png")