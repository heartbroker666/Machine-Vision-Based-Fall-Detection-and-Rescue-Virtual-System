"""
╔══════════════════════════════════════════════════════════════╗
║          pr2_rescue.py  ——  PR2 救援机器人控制器（最终版）        ║
╠══════════════════════════════════════════════════════════════╣
║  导航方案：A* 全局路径规划                                        ║
║    · 基于场景静态地图构建栅格地图                                   ║
║    · A* 算法预先规划绕障完整路径（主动绕开墙壁和家具）                 ║
║    · 路径简化：去除冗余中间点                                      ║
║    · 卡死检测 + 后退重新规划兜底                                   ║
╠══════════════════════════════════════════════════════════════╣
║  救援评价指标（每次救援到达后自动打印）：                              ║
║    · 响应时间   收到信号 → 开始移动（秒）                           ║
║    · 到达时间   收到信号 → 到达伤者（秒）                           ║
║    · 直线距离   起点到目标的直线距离（米）                            ║
║    · 实际路径   机器人实际行走总距离（米）                            ║
║    · 路径效率   直线距离 ÷ 实际路径（%，越接近100%越好）              ║
║    · 卡死次数   本次救援触发卡死脱困的次数                            ║
║    · 平均速度   实际路径 ÷ 到达时间（米/秒）                        ║
╚══════════════════════════════════════════════════════════════╝
"""

from controller import Robot
import math
import heapq
import time

# ═══════════════════════════════════════════════════════════════
#  系统参数（已根据实际场景微调）
# ═══════════════════════════════════════════════════════════════
TIME_STEP   = 32
MAX_SPEED   = 30.0
ARRIVE_DIST = 0.6    # 中间路径点到达判定距离（米）
FINAL_DIST  = 1.0    # 最终目标到达判定距离（米）

# A* 栅格地图
GRID_RES    = 0.1    # 栅格分辨率（米/格），越小越精细但计算越慢
ROBOT_CLEAR = 0.35   # 障碍物膨胀半径（米），保证机器人与障碍物的安全间距

# 场景边界（与 .wbt 场景一致，单位：米）
MAP_X_MIN, MAP_X_MAX = -9.9,  0.0
MAP_Y_MIN, MAP_Y_MAX = -6.6,  0.0

# 卡死检测
STUCK_INTERVAL  = 60    # 每60步（约2秒）检查一次位移
STUCK_THRESHOLD = 0.15  # 2秒内位移不足此值（米）则判定为卡死
BACKING_STEPS   = 80    # 后退持续步数（约2.5秒）
BACKING_SPEED   = 0.35  # 后退速度系数


# ═══════════════════════════════════════════════════════════════
#  场景障碍物地图
#  格式：(世界坐标X, 世界坐标Y, 障碍半径)
#  障碍半径 + ROBOT_CLEAR = A* 实际回避半径
# ═══════════════════════════════════════════════════════════════
OBSTACLES = [
    # ── 外围墙壁 ──────────────────────────────────────────────
    ( 0.00, -0.65, 0.5),   # wall1
    ( 0.00, -4.30, 0.5),   # wall2
    (-9.90, -2.30, 0.5),   # wall3
    (-3.80,  0.00, 0.6),   # wall5

    # ── wall9 隔断墙 ───────────────────────────────────────────
    # 物理参数：translation(-3.3, -1.8)，size(宽0.3 × 长3.3 × 高3.5)
    # 沿 Y 轴延伸，覆盖范围：y = -0.15 ~ -3.45
    # 每隔 0.3m 放置一个障碍点，确保 A* 完整感知整面墙
    (-3.30, -0.15, 0.5),
    (-3.30, -0.45, 0.5),
    (-3.30, -0.75, 0.5),
    (-3.30, -1.05, 0.5),
    (-3.30, -1.35, 0.5),
    (-3.30, -1.65, 0.5),
    (-3.30, -1.80, 0.5),   # 墙体中心
    (-3.30, -2.10, 0.5),
    (-3.30, -2.40, 0.5),
    (-3.30, -2.70, 0.5),
    (-3.30, -3.00, 0.5),
    (-3.30, -3.30, 0.5),
    (-3.30, -3.45, 0.5),

    # ── 餐桌椅组 ──────────────────────────────────────────────
    (-1.07, -4.94, 0.7),   # 餐桌
    (-1.46, -5.41, 0.4),   # 椅1
    (-0.64, -4.44, 0.4),   # 椅2
    (-1.39, -4.49, 0.4),   # 椅3
    (-0.71, -5.38, 0.4),   # 椅4

    # ── 书桌椅 ────────────────────────────────────────────────
    (-5.13, -0.51, 0.6),   # 书桌
    (-5.15, -0.89, 0.4),   # 木椅

    # ── 客厅沙发区 ────────────────────────────────────────────
    (-7.09, -2.55, 0.6),   # 沙发1
    (-7.09, -2.06, 0.6),   # 沙发2
    (-5.45, -2.51, 0.7),   # 扶手椅
    (-7.06, -0.80, 0.3),   # 茶几
    (-9.23, -2.92, 0.5),   # 沙发3

    # ── 厨房区 ────────────────────────────────────────────────
    (-0.52, -0.50, 0.4),   # 冰箱
    (-1.31, -0.15, 0.4),   # 橱柜1
    (-2.19, -0.15, 0.4),   # 橱柜2
    (-2.85, -0.58, 0.4),   # 烤箱

    # ── 书柜与植物 ────────────────────────────────────────────
    (-3.47, -6.43, 0.4),   # 书柜
    (-4.52, -6.08, 0.4),   # 盆栽
]


# ═══════════════════════════════════════════════════════════════
#  A* 栅格地图工具函数
# ═══════════════════════════════════════════════════════════════

def w2g(wx, wy):
    """世界坐标 → 栅格坐标"""
    gx = int((wx - MAP_X_MIN) / GRID_RES)
    gy = int((wy - MAP_Y_MIN) / GRID_RES)
    return gx, gy

def g2w(gx, gy):
    """栅格坐标 → 世界坐标（取格子中心）"""
    wx = gx * GRID_RES + MAP_X_MIN + GRID_RES / 2
    wy = gy * GRID_RES + MAP_Y_MIN + GRID_RES / 2
    return wx, wy

def build_grid():
    """构建二值占用栅格地图（0=自由，1=障碍）"""
    cols = int((MAP_X_MAX - MAP_X_MIN) / GRID_RES) + 2
    rows = int((MAP_Y_MAX - MAP_Y_MIN) / GRID_RES) + 2
    grid = [[0] * cols for _ in range(rows)]
    for (ox, oy, r) in OBSTACLES:
        inflate = r + ROBOT_CLEAR
        for gy in range(rows):
            for gx in range(cols):
                wx, wy = g2w(gx, gy)
                if math.sqrt((wx - ox)**2 + (wy - oy)**2) <= inflate:
                    grid[gy][gx] = 1
    return grid, rows, cols

def nearest_free(grid, rows, cols, gx, gy):
    """若指定格为障碍，向外搜索最近的空闲格"""
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
    """
    A* 路径规划
    返回世界坐标路径点列表，无路径时返回 None
    """
    sx, sy = w2g(*start_w)
    gx, gy = w2g(*goal_w)
    sx, sy = max(0, min(cols-1, sx)), max(0, min(rows-1, sy))
    gx, gy = max(0, min(cols-1, gx)), max(0, min(rows-1, gy))
    sx, sy = nearest_free(grid, rows, cols, sx, sy)
    gx, gy = nearest_free(grid, rows, cols, gx, gy)

    def h(x, y):
        return math.sqrt((x-gx)**2 + (y-gy)**2)

    open_set  = [(h(sx, sy), 0.0, sx, sy)]
    came_from = {}
    g_score   = {(sx, sy): 0.0}
    DIRS = [( 1, 0, 1.0  ), (-1, 0, 1.0  ), ( 0, 1, 1.0  ), ( 0,-1, 1.0  ),
            ( 1, 1, 1.414), ( 1,-1, 1.414), (-1, 1, 1.414), (-1,-1, 1.414)]

    while open_set:
        _, cost, cx, cy = heapq.heappop(open_set)
        if (cx, cy) in came_from and g_score.get((cx, cy), 1e9) < cost:
            continue
        if cx == gx and cy == gy:
            path = []
            node = (cx, cy)
            while node in came_from:
                path.append(g2w(*node))
                node = came_from[node]
            path.reverse()
            path.append(goal_w)   # 精确终点
            return path
        for dx, dy, step in DIRS:
            nx, ny = cx+dx, cy+dy
            if not (0 <= nx < cols and 0 <= ny < rows): continue
            if grid[ny][nx] == 1: continue
            ng = g_score.get((cx, cy), 1e9) + step
            if ng < g_score.get((nx, ny), 1e9):
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_set, (ng + h(nx, ny), ng, nx, ny))
    return None

def simplify(path, grid, rows, cols):
    """
    路径简化：若两点间直线无障碍则跳过中间点
    减少不必要的路径点，使行进路线更自然流畅
    """
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


# ═══════════════════════════════════════════════════════════════
#  PR2 救援控制器主类
# ═══════════════════════════════════════════════════════════════

class PR2Rescue(Robot):

    def __init__(self):
        super().__init__()

        # ── 驱动轮（8个全向轮） ───────────────────────────────
        wheel_names = [
            "fl_caster_l_wheel_joint", "fl_caster_r_wheel_joint",
            "fr_caster_l_wheel_joint", "fr_caster_r_wheel_joint",
            "bl_caster_l_wheel_joint", "bl_caster_r_wheel_joint",
            "br_caster_l_wheel_joint", "br_caster_r_wheel_joint",
        ]
        self.wheels = []
        for name in wheel_names:
            m = self.getDevice(name)
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
            self.wheels.append(m)

        # ── 转向关节（4个） ───────────────────────────────────
        rot_names = [
            "fl_caster_rotation_joint", "fr_caster_rotation_joint",
            "bl_caster_rotation_joint", "br_caster_rotation_joint",
        ]
        self.rot_motors  = [self.getDevice(n) for n in rot_names]
        self.rot_sensors = []
        for m in self.rot_motors:
            s = m.getPositionSensor()
            s.enable(TIME_STEP)
            self.rot_sensors.append(s)

        # ── 手臂关节 ──────────────────────────────────────────
        self.left_arm  = [self.getDevice(n) for n in [
            "l_shoulder_pan_joint", "l_shoulder_lift_joint",
            "l_upper_arm_roll_joint", "l_elbow_flex_joint", "l_wrist_roll_joint"]]
        self.right_arm = [self.getDevice(n) for n in [
            "r_shoulder_pan_joint", "r_shoulder_lift_joint",
            "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_wrist_roll_joint"]]
        self.torso  = self.getDevice("torso_lift_joint")
        torso_s     = self.getDevice("torso_lift_joint_sensor")
        if torso_s:
            torso_s.enable(TIME_STEP)

        # ── 头部关节 ──────────────────────────────────────────
        self.head_pan  = self.getDevice("head_pan_joint")
        self.head_tilt = self.getDevice("head_tilt_joint")

        # ── GPS ───────────────────────────────────────────────
        self.gps = self.getDevice("gps")
        if self.gps:
            self.gps.enable(TIME_STEP)
            print("[GPS] 已启用")
        else:
            print("[警告] 未找到 GPS")

        # ── Compass ───────────────────────────────────────────
        self.compass = self.getDevice("compass")
        if self.compass:
            self.compass.enable(TIME_STEP)
            print("[Compass] 已启用")
        else:
            print("[警告] 未找到 Compass")

        # ── Receiver ──────────────────────────────────────────
        self.receiver = self.getDevice("rescue_receiver")
        if self.receiver:
            self.receiver.enable(TIME_STEP)
            print("[Receiver] 已启用，监听 channel 1")
        else:
            print("[警告] 未找到 rescue_receiver")

        # ── 构建 A* 栅格地图（初始化时一次性完成）─────────────
        print("[A*] 构建栅格地图...")
        self.grid, self.grid_rows, self.grid_cols = build_grid()
        print(f"[A*] 完成 {self.grid_cols}×{self.grid_rows} 格，分辨率 {GRID_RES}m")

        # ── 导航状态 ──────────────────────────────────────────
        self.state        = "IDLE"    # IDLE / NAVIGATING / BACKING / ARRIVED
        self.target_x     = 0.0
        self.target_y     = 0.0
        self.path         = []
        self.path_idx     = 0
        self.rescue_count = 0

        # ── 卡死检测 ──────────────────────────────────────────
        self.stuck_steps   = 0
        self.stuck_pos     = (0.0, 0.0)
        self.backing_steps = 0
        self.backing_angle = 0.0

        # ── 救援评价数据 ──────────────────────────────────────
        self._stat        = {}    # 当次救援临时统计数据
        self.rescue_log   = []    # 所有历史救援记录

        print("[PR2] 初始化完成，待命中...\n")

    # ═══════════════════════════════════════════════════════════
    #  基础运动
    # ═══════════════════════════════════════════════════════════

    def set_speeds(self, l, r):
        l = max(-MAX_SPEED, min(MAX_SPEED, l))
        r = max(-MAX_SPEED, min(MAX_SPEED, r))
        for i in [0, 1, 4, 5]: self.wheels[i].setVelocity(l)
        for i in [2, 3, 6, 7]: self.wheels[i].setVelocity(r)

    def stop(self):
        self.set_speeds(0, 0)

    def move_toward(self, world_angle, speed):
        """朝世界坐标系指定角度移动"""
        heading = self.get_heading()
        rel     = world_angle - heading
        while rel >  math.pi: rel -= 2*math.pi
        while rel < -math.pi: rel += 2*math.pi
        for m in self.rot_motors:
            m.setPosition(rel)
        self.set_speeds(speed, speed)

    def set_arm_pose(self, pose="idle"):
        """
        idle   : 手臂收拢下垂（行进姿态）
        rescue : 双臂前伸（到达伤者后的救援姿态）
        """
        targets = [0.0, 1.35, 0.0, -1.57, 0.0] if pose == "idle" \
                  else [0.3, -0.1, 0.0, -0.5, 0.0]
        for i, t in enumerate(targets):
            self.left_arm[i].setPosition(t)
            self.right_arm[i].setPosition(-t if i == 0 else t)
        if self.torso:
            self.torso.setPosition(0.0 if pose == "idle" else 0.3)

    # ═══════════════════════════════════════════════════════════
    #  传感器读取
    # ═══════════════════════════════════════════════════════════

    def get_position(self):
        if not self.gps: return 0.0, 0.0
        v = self.gps.getValues()
        return v[0], v[1]

    def get_heading(self):
        if not self.compass: return 0.0
        v = self.compass.getValues()
        return math.atan2(v[0], v[1])

    # ═══════════════════════════════════════════════════════════
    #  A* 路径规划
    # ═══════════════════════════════════════════════════════════

    def plan_path(self):
        cx, cy = self.get_position()
        print(f"[A*] 规划: ({cx:.2f},{cy:.2f}) → ({self.target_x:.2f},{self.target_y:.2f})")

        path = astar(self.grid, self.grid_rows, self.grid_cols,
                     (cx, cy), (self.target_x, self.target_y))

        if path is None:
            # 规划失败：打印起点周围障碍地图辅助调试
            sx, sy = w2g(cx, cy)
            sx = max(0, min(self.grid_cols-1, sx))
            sy = max(0, min(self.grid_rows-1, sy))
            print(f"[A*] ⚠ 无路径！起点格({sx},{sy})={self.grid[sy][sx]}")
            print("[A*] 起点周围障碍情况（█=障碍 ·=空闲）：")
            for dy in range(-3, 4):
                row = ""
                for dx in range(-3, 4):
                    nx, ny = sx+dx, sy+dy
                    if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                        row += "█" if self.grid[ny][nx] else "·"
                    else:
                        row += "X"
                center = " ←起点" if dy == 0 else ""
                print(f"  {row}{center}")
            self.path = [(self.target_x, self.target_y)]
        else:
            path = simplify(path, self.grid, self.grid_rows, self.grid_cols)
            path[-1] = (self.target_x, self.target_y)
            self.path = path
            print(f"[A*] 规划完成，共 {len(path)} 个路径点：")
            for i, (px, py) in enumerate(path):
                flag = " ← 终点" if i == len(path)-1 else ""
                print(f"     [{i}] ({px:.2f}, {py:.2f}){flag}")

        self.path_idx = 0

    # ═══════════════════════════════════════════════════════════
    #  路径跟踪（含行走距离累计）
    # ═══════════════════════════════════════════════════════════

    def follow_path(self):
        """
        逐点跟踪 A* 路径
        同步累计实际行走距离，供救援评价使用
        返回 True 表示已到达最终目标
        """
        if not self.path:
            return True

        cx, cy     = self.get_position()
        wp_x, wp_y = self.path[self.path_idx]
        dx   = wp_x - cx
        dy   = wp_y - cy
        dist = math.sqrt(dx**2 + dy**2)

        is_last   = (self.path_idx == len(self.path) - 1)
        threshold = FINAL_DIST if is_last else ARRIVE_DIST

        # 累计实际行走距离
        last = self._stat.get("last_pos")
        if last is not None:
            self._stat["path_length"] += math.sqrt((cx-last[0])**2 + (cy-last[1])**2)
        self._stat["last_pos"] = (cx, cy)

        if dist < threshold:
            if is_last:
                self.stop()
                return True
            self.path_idx += 1
            print(f"[路径] ✓ 路径点 {self.path_idx-1} → {self.path_idx}")
            wp_x, wp_y = self.path[self.path_idx]
            dx   = wp_x - cx
            dy   = wp_y - cy
            dist = math.sqrt(dx**2 + dy**2)

        # 头部始终朝向最终目标
        fx, fy   = self.path[-1]
        heading  = self.get_heading()
        head_rel = math.atan2(fy-cy, fx-cx) - heading
        while head_rel >  math.pi: head_rel -= 2*math.pi
        while head_rel < -math.pi: head_rel += 2*math.pi
        if self.head_pan:
            self.head_pan.setPosition(max(-2.8, min(2.8, head_rel)))
        if self.head_tilt:
            self.head_tilt.setPosition(0.3 if dist > 2.0 else 0.6)

        speed = max(MAX_SPEED * 0.4, MAX_SPEED * min(1.0, dist / 2.0))
        self.move_toward(math.atan2(dy, dx), speed)
        return False

    # ═══════════════════════════════════════════════════════════
    #  救援评价数据统计
    # ═══════════════════════════════════════════════════════════

    def _stat_start(self):
        """收到救援信号时初始化本次统计"""
        cx, cy = self.get_position()
        straight = math.sqrt((cx-self.target_x)**2 + (cy-self.target_y)**2)
        self._stat = {
            "no"           : self.rescue_count,
            "t_signal"     : time.time(),   # 收到信号的时间戳
            "t_move"       : None,          # 开始移动的时间戳
            "start_pos"    : (cx, cy),
            "straight_dist": round(straight, 3),
            "path_length"  : 0.0,
            "stuck_count"  : 0,
            "last_pos"     : (cx, cy),
        }

    def _stat_check_move(self):
        """检测是否已离开起点，记录开始移动时间（仅记录一次）"""
        if self._stat.get("t_move") is not None:
            return
        cx, cy = self.get_position()
        sx, sy = self._stat["start_pos"]
        if math.sqrt((cx-sx)**2 + (cy-sy)**2) > 0.1:
            self._stat["t_move"] = time.time()

    def _stat_finish(self):
        """到达目标后计算并打印所有评价指标"""
        s  = self._stat
        if not s:
            return

        t_arrive  = time.time()
        t_signal  = s["t_signal"]
        t_move    = s["t_move"] if s["t_move"] else t_arrive
        straight  = s["straight_dist"]
        path_len  = round(s["path_length"], 2)
        stuck_n   = s["stuck_count"]
        total_t   = round(t_arrive - t_signal, 2)
        resp_t    = round(t_move   - t_signal, 2)
        efficiency= round(min(straight / path_len, 1.0) * 100, 1) if path_len > 0.01 else 100.0
        avg_speed = round(path_len / total_t, 3) if total_t > 0 else 0.0

        record = {
            "no"          : s["no"],
            "response_t"  : resp_t,
            "total_t"     : total_t,
            "straight"    : straight,
            "path_len"    : path_len,
            "efficiency"  : efficiency,
            "avg_speed"   : avg_speed,
            "stuck_count" : stuck_n,
        }
        self.rescue_log.append(record)

        # ── 单次救援评价打印 ──────────────────────────────────
        W = 52
        print("\n" + "═"*W)
        print(f"{'第 '+str(s['no'])+' 次救援完成 —— 效果评价':^{W}}")
        print("═"*W)
        print(f"  {'响应时间':<10} {resp_t:>7.2f} 秒   收到信号 → 开始移动")
        print(f"  {'到达时间':<10} {total_t:>7.2f} 秒   收到信号 → 到达伤者")
        print(f"  {'直线距离':<10} {straight:>7.2f} 米")
        print(f"  {'实际路径':<10} {path_len:>7.2f} 米")
        print(f"  {'路径效率':<10} {efficiency:>7.1f} %    越接近100%越好")
        print(f"  {'平均速度':<10} {avg_speed:>7.3f} 米/秒")
        print(f"  {'卡死次数':<10} {stuck_n:>7} 次")
        print("═"*W)

        # ── 多次救援汇总（≥2次时打印）────────────────────────
        if len(self.rescue_log) >= 2:
            n         = len(self.rescue_log)
            avg_total = round(sum(r["total_t"]    for r in self.rescue_log) / n, 2)
            avg_resp  = round(sum(r["response_t"] for r in self.rescue_log) / n, 2)
            avg_eff   = round(sum(r["efficiency"] for r in self.rescue_log) / n, 1)
            avg_spd   = round(sum(r["avg_speed"]  for r in self.rescue_log) / n, 3)
            tot_stuck = sum(r["stuck_count"] for r in self.rescue_log)
            print(f"\n  ┌{'─'*(W-4)}┐")
            print(f"  │{'【'+str(n)+'次救援汇总统计】':^{W-4}}│")
            print(f"  ├{'─'*(W-4)}┤")
            print(f"  │  {'平均响应时间':<12} {avg_resp:>6.2f} 秒{'':<18}│")
            print(f"  │  {'平均到达时间':<12} {avg_total:>6.2f} 秒{'':<18}│")
            print(f"  │  {'平均路径效率':<12} {avg_eff:>6.1f} %{'':<19}│")
            print(f"  │  {'平均移动速度':<12} {avg_spd:>6.3f} 米/秒{'':<15}│")
            print(f"  │  {'累计卡死次数':<12} {tot_stuck:>6} 次{'':<19}│")
            print(f"  └{'─'*(W-4)}┘\n")

    # ═══════════════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════════════

    def run(self):
        self.set_arm_pose("idle")
        if self.head_pan:  self.head_pan.setPosition(0.0)
        if self.head_tilt: self.head_tilt.setPosition(0.0)

        # 等待传感器数据稳定
        for _ in range(50):
            self.step(TIME_STEP)

        print(f"[PR2] 初始朝向: {math.degrees(self.get_heading()):.1f}°\n")

        while self.step(TIME_STEP) != -1:

            # ── 接收跌倒坐标（Emitter → Receiver）──────────────
            while self.receiver and self.receiver.getQueueLength() > 0:
                try:
                    msg  = self.receiver.getString()
                    self.receiver.nextPacket()
                    x, y = map(float, msg.strip().split(","))

                    if self.state in ("IDLE", "ARRIVED"):
                        self.target_x      = x
                        self.target_y      = y
                        self.rescue_count += 1
                        self.stuck_steps   = 0
                        self.backing_steps = 0
                        self.stuck_pos     = self.get_position()
                        self.set_arm_pose("idle")
                        self.plan_path()
                        self._stat_start()        # 开始计时
                        self.state = "NAVIGATING"
                        print(f"[PR2] 收到救援信号 #{self.rescue_count}，"
                              f"目标=({x:.2f},{y:.2f})")

                except Exception as e:
                    print(f"[PR2] 解析出错: {e}")
                    if self.receiver.getQueueLength() > 0:
                        self.receiver.nextPacket()

            # ── 状态机 ──────────────────────────────────────────
            if self.state == "NAVIGATING":

                # 检测是否已开始移动
                self._stat_check_move()

                # 卡死检测（每 STUCK_INTERVAL 步检查一次）
                self.stuck_steps += 1
                if self.stuck_steps >= STUCK_INTERVAL:
                    cx, cy = self.get_position()
                    px, py = self.stuck_pos
                    if math.sqrt((cx-px)**2 + (cy-py)**2) < STUCK_THRESHOLD:
                        self._stat["stuck_count"] += 1
                        self.backing_angle = math.atan2(py-cy, px-cx)
                        self.backing_steps = 0
                        self.state         = "BACKING"
                        print(f"[PR2] ⚠ 卡死（第 {self._stat['stuck_count']} 次），后退脱困")
                    else:
                        self.stuck_pos   = (cx, cy)
                        self.stuck_steps = 0

                # 路径跟踪
                if self.state == "NAVIGATING":
                    if self.follow_path():
                        self.state = "ARRIVED"
                        self.set_arm_pose("rescue")
                        if self.head_tilt:
                            self.head_tilt.setPosition(0.8)
                        cx, cy = self.get_position()
                        print(f"[PR2] ✅ 已到达伤者位置！当前坐标=({cx:.2f},{cy:.2f})")
                        self._stat_finish()       # 打印评价数据

            elif self.state == "BACKING":
                self.backing_steps += 1
                self.move_toward(self.backing_angle, MAX_SPEED * BACKING_SPEED)
                if self.backing_steps >= BACKING_STEPS:
                    self.stop()
                    self.plan_path()
                    self.state       = "NAVIGATING"
                    self.stuck_steps = 0
                    self.stuck_pos   = self.get_position()
                    print("[PR2] 后退完成，重新规划路径")

            elif self.state == "ARRIVED":
                self.stop()

            elif self.state == "IDLE":
                self.stop()


controller = PR2Rescue()
controller.run()