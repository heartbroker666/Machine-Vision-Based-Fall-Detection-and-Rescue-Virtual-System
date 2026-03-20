"""
fall_detection_ui.py
跌倒检测系统 —— 图形化启动界面
功能：
  · 系统启动 / 停止控制
  · 实时参数配置（模型路径、摄像头、阈值等）
  · 运行状态监控（报警次数、帧数、运行时长）
  · 跌倒事件日志列表
  · 录像回放（双击日志条目回放对应片段）
依赖：tkinter（内置）、opencv-python、pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import os
import csv
import time
import cv2
from datetime import datetime
from PIL import Image, ImageTk

# ══════════════════════════════════════════════════════
#  配色与字体
# ══════════════════════════════════════════════════════
BG_DARK   = "#0f1117"
BG_PANEL  = "#181c27"
BG_CARD   = "#1e2333"
BG_HOVER  = "#252a3d"
ACCENT    = "#4f8ef7"
ACCENT2   = "#00d4aa"
WARN      = "#ff6b35"
DANGER    = "#ff3b5c"
SUCCESS   = "#2ecc71"
TEXT_PRI  = "#e8eaf0"
TEXT_SEC  = "#8b92a8"
TEXT_DIM  = "#4a5068"
BORDER    = "#2a3050"

FONT_TITLE = ("Microsoft YaHei UI", 13, "bold")
FONT_LABEL = ("Microsoft YaHei UI", 10)
FONT_SMALL = ("Microsoft YaHei UI", 9)
FONT_MONO  = ("Consolas", 9)
FONT_BIG   = ("Microsoft YaHei UI", 22, "bold")
FONT_MED   = ("Microsoft YaHei UI", 14, "bold")

OUTPUT_DIR = "D:/Users/Lenovo/Graduation_project/detection_output"


# ══════════════════════════════════════════════════════
#  圆角按钮（用 Frame + Label 实现，避免 Canvas 尺寸冲突）
# ══════════════════════════════════════════════════════
class RoundButton(tk.Frame):
    def __init__(self, parent, text, command=None,
                 width=120, height=36, bg=ACCENT, fg=TEXT_PRI,
                 radius=8, font=FONT_LABEL, **kw):
        # 过滤掉不属于 Frame 的参数
        kw.pop("radius", None)
        super().__init__(parent, bg=bg,
                         width=width, height=height,
                         highlightthickness=0, **kw)
        self.pack_propagate(False)   # 保持固定尺寸
        self._bg  = bg
        self._fg  = fg
        self._cmd = command

        self._lbl = tk.Label(self, text=text, font=font,
                             bg=bg, fg=fg, cursor="hand2",
                             padx=8, pady=0)
        self._lbl.place(relx=0.5, rely=0.5, anchor="center")

        for w in (self, self._lbl):
            w.bind("<Button-1>", lambda _: command() if command else None)
            w.bind("<Enter>",    lambda _: self._hover(True))
            w.bind("<Leave>",    lambda _: self._hover(False))

    def _lighten(self, hex_color):
        import colorsys
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
        hh, s, v = colorsys.rgb_to_hsv(r, g, b)
        r2, g2, b2 = colorsys.hsv_to_rgb(hh, s, min(v*1.22, 1.0))
        return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))

    def _hover(self, on):
        c = self._lighten(self._bg) if on else self._bg
        self.configure(bg=c)
        self._lbl.configure(bg=c)

    def update_text(self, text, bg=None):
        self._lbl.configure(text=text)
        if bg:
            self._bg = bg
            self.configure(bg=bg)
            self._lbl.configure(bg=bg)


# ══════════════════════════════════════════════════════
#  录像回放窗口
# ══════════════════════════════════════════════════════
class PlaybackWindow(tk.Toplevel):
    def __init__(self, parent, video_path):
        super().__init__(parent)
        self.title("跌倒录像回放")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.grab_set()

        self._path    = video_path
        self._cap     = None
        self._playing = False
        self._after   = None
        self._pos     = 0
        self._total   = 0
        self._seeking = False

        self._build()
        self._open()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self):
        # 标题栏
        bar = tk.Frame(self, bg=BG_PANEL, height=44)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="⏵  录像回放", font=FONT_MED,
                 bg=BG_PANEL, fg=TEXT_PRI).pack(side="left", padx=16, pady=8)
        tk.Label(bar, text=os.path.basename(self._path),
                 font=FONT_SMALL, bg=BG_PANEL, fg=TEXT_SEC).pack(side="left", padx=4)
        tk.Button(bar, text="✕", font=FONT_SMALL, bg=BG_PANEL,
                  fg=TEXT_SEC, relief="flat", bd=0,
                  command=self._close).pack(side="right", padx=12)

        # 画面
        self._canvas = tk.Canvas(self, width=640, height=400,
                                  bg="#000000", highlightthickness=0)
        self._canvas.pack(padx=16, pady=(12, 4))

        # 进度条 + 时间
        pf = tk.Frame(self, bg=BG_DARK)
        pf.pack(fill="x", padx=16, pady=(0, 2))

        self._time_var = tk.StringVar(value="00:00 / 00:00")
        tk.Label(pf, textvariable=self._time_var, font=FONT_MONO,
                 bg=BG_DARK, fg=TEXT_SEC).pack(side="right")

        self._prog_var = tk.DoubleVar(value=0)
        self._prog = ttk.Scale(pf, variable=self._prog_var,
                                from_=0, to=100, orient="horizontal",
                                command=self._on_seek)
        self._prog.pack(side="left", fill="x", expand=True, padx=(0,10))

        # 控制按钮
        ctrl = tk.Frame(self, bg=BG_DARK)
        ctrl.pack(pady=10)

        self._btn = RoundButton(ctrl, "▶  播放", self._toggle,
                                 width=110, height=36, bg=ACCENT)
        self._btn.pack(side="left", padx=6)
        RoundButton(ctrl, "⏮  重置", self._restart,
                    width=90, height=36, bg=BG_CARD).pack(side="left", padx=6)
        RoundButton(ctrl, "✕  关闭", self._close,
                    width=90, height=36, bg=BG_CARD).pack(side="left", padx=6)

    def _open(self):
        if not os.path.exists(self._path):
            messagebox.showerror("错误", f"视频文件不存在：\n{self._path}", parent=self)
            self.destroy()
            return
        self._cap   = cv2.VideoCapture(self._path)
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps   = self._cap.get(cv2.CAP_PROP_FPS) or 25
        self._show()

    def _show(self):
        if not self._cap:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._playing = False
            self._btn.update_text("▶  播放", ACCENT)
            return
        self._pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.resize(frame, (640, 400))
        img   = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self._canvas.create_image(0, 0, anchor="nw", image=img)
        self._canvas._img = img

        if not self._seeking and self._total > 0:
            self._prog_var.set(self._pos / self._total * 100)

        cur_s = int(self._pos / self._fps)
        tot_s = int(self._total / self._fps)
        self._time_var.set(f"{cur_s//60:02d}:{cur_s%60:02d} / "
                            f"{tot_s//60:02d}:{tot_s%60:02d}")

    def _toggle(self):
        self._playing = not self._playing
        if self._playing:
            self._btn.update_text("⏸  暂停", WARN)
            self._loop()
        else:
            self._btn.update_text("▶  播放", ACCENT)
            if self._after:
                self.after_cancel(self._after)

    def _loop(self):
        if not self._playing:
            return
        self._show()
        if self._pos < self._total:
            self._after = self.after(int(1000 / self._fps), self._loop)
        else:
            self._playing = False
            self._btn.update_text("▶  播放", ACCENT)

    def _on_seek(self, val):
        if self._total > 0 and self._cap:
            self._seeking = True
            idx = int(float(val) / 100 * self._total)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._show()
            self._seeking = False

    def _restart(self):
        if self._after:
            self.after_cancel(self._after)
        self._playing = False
        self._btn.update_text("▶  播放", ACCENT)
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._show()

    def _close(self):
        self._playing = False
        if self._after:
            self.after_cancel(self._after)
        if self._cap:
            self._cap.release()
        self.destroy()


# ══════════════════════════════════════════════════════
#  主应用
# ══════════════════════════════════════════════════════
class FallDetectionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("跌倒检测救援系统  ——  控制台")
        self.configure(bg=BG_DARK)
        self.geometry("980x700")
        self.minsize(860, 600)

        self._running     = False
        self._proc        = None   # 检测子进程句柄
        self._start_time  = None
        self._alarm_count = 0
        self._frame_count = 0
        self._log_entries = []

        self._style()
        self._build()
        self._tick()

    # ── ttk 样式 ──
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Treeview",
                     background=BG_CARD, fieldbackground=BG_CARD,
                     foreground=TEXT_PRI, rowheight=28, borderwidth=0,
                     font=FONT_SMALL)
        s.configure("Treeview.Heading",
                     background=BG_PANEL, foreground=TEXT_SEC,
                     relief="flat", font=FONT_SMALL)
        s.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
        s.configure("Vertical.TScrollbar",
                     background=BG_CARD, troughcolor=BG_DARK, arrowcolor=TEXT_SEC)

    # ── 整体布局 ──
    def _build(self):
        # 顶栏
        topbar = tk.Frame(self, bg=BG_PANEL, height=54)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="◉", font=("Consolas", 18),
                 bg=BG_PANEL, fg=ACCENT).pack(side="left", padx=(18,6), pady=10)
        tk.Label(topbar, text="跌倒检测救援系统",
                 font=FONT_TITLE, bg=BG_PANEL, fg=TEXT_PRI).pack(side="left", pady=10)
        tk.Label(topbar, text="Fall Detection & Rescue Console",
                 font=FONT_SMALL, bg=BG_PANEL, fg=TEXT_DIM).pack(side="left", padx=10)

        self._dot = tk.Label(topbar, text="●", font=("Consolas", 16),
                              bg=BG_PANEL, fg=TEXT_DIM)
        self._dot.pack(side="right", padx=(0,10))
        self._status_lbl = tk.Label(topbar, text="待机",
                                     font=FONT_LABEL, bg=BG_PANEL, fg=TEXT_SEC)
        self._status_lbl.pack(side="right", padx=(0,4))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # 主体
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG_PANEL, width=282)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_left(left)

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)
        self._build_right(right)

    # ── 左侧配置 ──
    def _build_left(self, p):
        def sep():
            tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(4,0))

        def section(title):
            f = tk.Frame(p, bg=BG_PANEL)
            f.pack(fill="x", padx=14, pady=(12,3))
            tk.Label(f, text=title, font=("Microsoft YaHei UI", 9, "bold"),
                     bg=BG_PANEL, fg=ACCENT).pack(anchor="w")
            sep()

        def row(label, var_name, default, browse_file=False):
            f = tk.Frame(p, bg=BG_PANEL)
            f.pack(fill="x", padx=14, pady=3)
            tk.Label(f, text=label, font=FONT_SMALL, bg=BG_PANEL,
                     fg=TEXT_SEC, width=9, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default))
            setattr(self, var_name, var)
            e = tk.Entry(f, textvariable=var, font=FONT_MONO,
                         bg=BG_CARD, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                         relief="flat", bd=4)
            e.pack(side="left", fill="x", expand=True)
            if browse_file:
                tk.Button(f, text="…", font=FONT_SMALL, bg=BG_HOVER,
                          fg=TEXT_SEC, relief="flat", bd=0,
                          command=lambda: var.set(
                              filedialog.askopenfilename(
                                  filetypes=[("模型", "*.pt"), ("所有", "*.*")])
                              or var.get()
                          )).pack(side="left", padx=2)

        tk.Label(p, text="系统配置", font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_SEC).pack(anchor="w", padx=16, pady=(16,4))
        sep()

        section("📁  路径设置")
        row("检测脚本", "_v_script",
            "D:/Users/software/Webots/Webots/lib/controller/python/Webots/Driver/Fall_detection_final.py",
            browse_file=True)
        row("模型路径", "_v_model",
            "D:/pythonProject/毕设3/runs/detect/fall_mixed_v1/weights/best.pt",
            browse_file=True)
        row("摄像头名", "_v_cam",     "fist_cam")
        row("Emitter", "_v_emitter", "fall_emitter")
        row("输出目录", "_v_outdir",  OUTPUT_DIR)

        section("⚙  检测参数")

        # 置信度滑块
        sf = tk.Frame(p, bg=BG_PANEL)
        sf.pack(fill="x", padx=14, pady=3)
        tk.Label(sf, text="置信度", font=FONT_SMALL, bg=BG_PANEL,
                 fg=TEXT_SEC, width=9, anchor="w").pack(side="left")
        self._v_conf   = tk.DoubleVar(value=0.5)
        self._lbl_conf = tk.Label(sf, text="0.50", font=FONT_MONO,
                                   bg=BG_PANEL, fg=ACCENT, width=4)
        self._lbl_conf.pack(side="right")
        tk.Scale(sf, variable=self._v_conf, from_=0.1, to=0.95,
                 resolution=0.05, orient="horizontal",
                 bg=BG_PANEL, fg=TEXT_PRI, troughcolor=BG_CARD,
                 highlightthickness=0, showvalue=False, bd=0,
                 command=lambda v: self._lbl_conf.config(
                     text=f"{float(v):.2f}")).pack(side="left", fill="x", expand=True)

        row("确认帧数",    "_v_frames",   "15")
        row("冷却时间(s)", "_v_cooldown", "8.0")

        # 按钮区
        tk.Frame(p, bg=BG_PANEL, height=14).pack()
        sep()
        tk.Frame(p, bg=BG_PANEL, height=10).pack()

        bf = tk.Frame(p, bg=BG_PANEL)
        bf.pack(fill="x", padx=14)

        self._btn_run = RoundButton(bf, "▶  启动检测", self._toggle_run,
                                     width=250, height=42, bg=SUCCESS,
                                     font=FONT_MED)
        self._btn_run.pack(pady=3)

        RoundButton(bf, "📂  打开输出目录", self._open_dir,
                    width=250, height=34, bg=BG_CARD,
                    font=FONT_LABEL).pack(pady=2)

        RoundButton(bf, "🗑  清空事件列表", self._clear_log,
                    width=250, height=34, bg=BG_CARD,
                    font=FONT_LABEL).pack(pady=2)

    # ── 右侧监控 ──
    def _build_right(self, p):
        # 统计卡片
        cards = tk.Frame(p, bg=BG_DARK)
        cards.pack(fill="x", padx=14, pady=(14,8))

        def card(master, title, attr, color, unit=""):
            c = tk.Frame(master, bg=BG_CARD, padx=14, pady=10)
            c.pack(side="left", expand=True, fill="both", padx=4)
            tk.Label(c, text=title, font=FONT_SMALL,
                     bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w")
            var = tk.StringVar(value="0")
            setattr(self, attr, var)
            tk.Label(c, textvariable=var, font=FONT_BIG,
                     bg=BG_CARD, fg=color).pack(anchor="w")
            if unit:
                tk.Label(c, text=unit, font=FONT_SMALL,
                         bg=BG_CARD, fg=TEXT_DIM).pack(anchor="e")

        card(cards, "报警次数",   "_sv_alarm",   DANGER,  "次")
        card(cards, "已处理帧数", "_sv_frames",  ACCENT,  "帧")
        card(cards, "运行时长",   "_sv_runtime", ACCENT2, "")
        card(cards, "当前状态",   "_sv_state",   SUCCESS, "")

        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=14)

        # 日志标题
        hdr = tk.Frame(p, bg=BG_DARK)
        hdr.pack(fill="x", padx=14, pady=(10,4))
        tk.Label(hdr, text="跌倒事件记录", font=FONT_LABEL,
                 bg=BG_DARK, fg=TEXT_PRI).pack(side="left")
        tk.Label(hdr, text="双击任意行 → 回放录像",
                 font=FONT_SMALL, bg=BG_DARK, fg=TEXT_DIM).pack(side="right")

        # 日志表格
        tf = tk.Frame(p, bg=BG_DARK)
        tf.pack(fill="both", expand=True, padx=14, pady=(0,0))

        cols = ("#", "时间", "置信度", "位置 X", "位置 Y", "帧号", "视频文件")
        self._tree = ttk.Treeview(tf, columns=cols,
                                   show="headings", selectmode="browse")
        for c, w in zip(cols, [36, 76, 64, 70, 70, 58, 220]):
            self._tree.heading(c, text=c)
            self._tree.column(c, width=w, anchor="center", minwidth=28)
        self._tree.column("视频文件", anchor="w")

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._tree.tag_configure("odd",  background=BG_CARD)
        self._tree.tag_configure("even", background=BG_PANEL)
        self._tree.tag_configure("new",  background="#1a2e1a", foreground=SUCCESS)
        self._tree.bind("<Double-1>", self._dblclick)

        # 底部状态栏
        bar = tk.Frame(p, bg=BG_PANEL, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._bar_lbl = tk.Label(bar, text="就绪",
                                  font=FONT_MONO, bg=BG_PANEL, fg=TEXT_DIM)
        self._bar_lbl.pack(side="left", padx=12)
        tk.Label(bar, text="v1.0  |  双击日志行回放录像",
                 font=FONT_MONO, bg=BG_PANEL, fg=TEXT_DIM).pack(side="right", padx=12)

    # ──────────────────────────────────────────
    #  交互逻辑
    # ──────────────────────────────────────────
    def _toggle_run(self):
        if not self._running:
            self._start()
        else:
            self._stop()

    def _start(self):
        # ── 检查检测脚本路径 ──────────────────
        script_path = self._v_script.get().strip()
        if not os.path.exists(script_path):
            messagebox.showerror(
                "路径错误",
                f"找不到检测脚本：\n{script_path}\n\n"
                "请在配置面板中设置正确的脚本路径。",
                parent=self)
            return

        # ── 把UI参数写入环境变量 ──────────────
        import copy, sys
        env = copy.copy(os.environ)
        env.update({
            "FD_MODEL":    self._v_model.get(),
            "FD_CAM":      self._v_cam.get(),
            "FD_EMITTER":  self._v_emitter.get(),
            "FD_OUTDIR":   self._v_outdir.get(),
            "FD_CONF":     f"{self._v_conf.get():.2f}",
            "FD_FRAMES":   self._v_frames.get(),
            "FD_COOLDOWN": self._v_cooldown.get(),
        })

        # ── 启动检测子进程 ────────────────────
        try:
            self._proc = subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NEW_CONSOLE
                    if os.name == "nt" else 0
            )
        except Exception as e:
            messagebox.showerror("启动失败", f"无法启动检测脚本：\n{e}", parent=self)
            return

        self._running    = True
        self._start_time = time.time()
        self._btn_run.update_text("⏹  停止检测", DANGER)
        self._dot.config(fg=SUCCESS)
        self._status_lbl.config(text="运行中", fg=SUCCESS)
        self._sv_state.set("运行中")
        self._bar_lbl.config(text="检测进程已启动，等待 Webots 连接…")

        # 后台线程：监控进程输出 + 轮询日志
        threading.Thread(target=self._watch_proc, daemon=True).start()
        threading.Thread(target=self._poll_log,   daemon=True).start()

    def _stop(self):
        self._running = False
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._btn_run.update_text("▶  启动检测", SUCCESS)
        self._dot.config(fg=TEXT_DIM)
        self._status_lbl.config(text="已停止", fg=TEXT_SEC)
        self._sv_state.set("已停止")
        self._bar_lbl.config(text="检测已停止")

    def _open_dir(self):
        d = self._v_outdir.get()
        if os.path.exists(d):
            os.startfile(d)
        else:
            messagebox.showinfo("提示", f"目录不存在：\n{d}", parent=self)

    def _clear_log(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._log_entries.clear()
        self._alarm_count = 0
        self._sv_alarm.set("0")

    def _watch_proc(self):
        """监控子进程的控制台输出，进程退出后更新UI状态"""
        if not self._proc:
            return
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                self.after(0, self._bar_lbl.config, {"text": line[:80]})
        # 进程结束
        ret = self._proc.wait()
        if self._running:
            msg = f"检测进程已退出（返回码 {ret}）"
            self.after(0, self._stop)
            self.after(0, self._bar_lbl.config, {"text": msg})

    def _poll_log(self):
        """后台轮询最新 CSV 日志，将新行推送到表格"""
        outdir    = self._v_outdir.get()
        last_file = None
        last_line = 0

        while self._running:
            time.sleep(1.5)
            try:
                if not os.path.exists(outdir):
                    continue
                csvs = sorted(
                    [f for f in os.listdir(outdir)
                     if f.startswith("fall_log") and f.endswith(".csv")],
                    reverse=True)
                if not csvs:
                    continue

                newest = os.path.join(outdir, csvs[0])
                if newest != last_file:
                    last_file = newest
                    last_line = 0

                with open(newest, "r", encoding="utf-8-sig") as f:
                    rows = list(csv.reader(f))

                vid_name = csvs[0].replace("fall_log_", "demo_").replace(".csv", ".mp4")
                vid_path = os.path.join(outdir, vid_name)

                new_rows = rows[last_line+1:]
                for row in new_rows:
                    if len(row) >= 6:
                        self._alarm_count += 1
                        entry = tuple(row[:6]) + (vid_path,)
                        self._log_entries.append(entry)
                        self.after(0, self._insert_row, entry)
                last_line = len(rows) - 1

                if len(rows) > 1:
                    try:
                        self._frame_count = int(rows[-1][5])
                    except Exception:
                        pass
            except Exception:
                pass

    def _insert_row(self, entry):
        n   = len(self._tree.get_children())
        tag = "new" if n == 0 else ("odd" if n % 2 else "even")
        iid = self._tree.insert("", "end", values=entry, tags=(tag,))
        self._tree.see(iid)
        self._sv_alarm.set(str(self._alarm_count))

    def _dblclick(self, event):
        sel = self._tree.selection()
        if not sel:
            return
        vals = self._tree.item(sel[0], "values")
        if len(vals) < 7:
            return
        vid_path = vals[6]
        if not os.path.exists(vid_path):
            vid_path = os.path.join(self._v_outdir.get(), os.path.basename(vid_path))
        PlaybackWindow(self, vid_path)

    def _tick(self):
        if self._running and self._start_time:
            e = int(time.time() - self._start_time)
            self._sv_runtime.set(f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")
            self._sv_frames.set(str(self._frame_count))
        self.after(1000, self._tick)

    def load_history(self):
        """启动时加载历史日志（最近5次）"""
        outdir = self._v_outdir.get()
        if not os.path.exists(outdir):
            return
        csvs = sorted(
            [f for f in os.listdir(outdir)
             if f.startswith("fall_log") and f.endswith(".csv")],
            reverse=True)[:5]
        for csv_name in csvs:
            vid_name = csv_name.replace("fall_log_", "demo_").replace(".csv", ".mp4")
            vid_path = os.path.join(outdir, vid_name)
            try:
                with open(os.path.join(outdir, csv_name), "r", encoding="utf-8-sig") as f:
                    for row in list(csv.reader(f))[1:]:
                        if len(row) >= 6:
                            entry = tuple(row[:6]) + (vid_path,)
                            self._log_entries.append(entry)
                            n   = len(self._tree.get_children())
                            tag = "odd" if n % 2 else "even"
                            self._tree.insert("", "end", values=entry, tags=(tag,))
                            self._alarm_count += 1
            except Exception:
                pass
        if self._alarm_count:
            self._sv_alarm.set(str(self._alarm_count))
            self._bar_lbl.config(
                text=f"已加载历史日志 {len(csvs)} 个文件，共 {self._alarm_count} 条记录")


# ══════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    app = FallDetectionApp()
    app.after(400, app.load_history)
    app.mainloop()