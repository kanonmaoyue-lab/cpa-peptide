"""cpamp_gui.py -- 环肽双通道预测（图形界面 v3）。

═══ 设计要点 ═══
    · 首页三大功能用**渐变按钮**区分（蓝=预测 / 橙=标注学习 / 绿=诊断偏差）
    · 背景很浅的蓝（#F2F7FC），卡片纯白
    · 每个功能独立成页，左上角可返回；每页都有「清空」
    · 菌株输入支持**输入前缀筛选**（下拉里是训练集的 155 株）
    · 顶部「帮助」打开使用教程（何时该做标注学习 / 诊断）
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpamp_lib"))

ASSETS = ROOT / "assets"
DATA = ROOT / "data"
MODELS = ROOT / "models"

# ── 配色 ──
BG = "#F2F7FC"
CARD = "#FFFFFF"
LINE = "#DCE7F3"
INK = "#1B3A57"
INK2 = "#5B7A99"
HI = "#2E6EB2"

# ── 接种规格（cfuGroup 0–4）──
CFU_OPTS = ["（未提供）", "低（<1E4）", "常规偏低（1E4–1E5）",
            "标准（1E5–1E6）", "偏高（1E6–1E7）", "高（>1E7）"]
CFU_VAL = {CFU_OPTS[0]: None, CFU_OPTS[1]: 0, CFU_OPTS[2]: 1,
           CFU_OPTS[3]: 2, CFU_OPTS[4]: 3, CFU_OPTS[5]: 4}
MED_OPTS = ["（未提供）", "MHB", "CAMHB", "LBB", "TSB", "RPMI-1640"]

SAMPLE = """# 每行一条，格式：ID,SMILES[,序列]（可从 Excel 直接复制三列）
P1,NC(=O)C[C@@H]1NC(=O)[C@@H](NC(=O)[C@@H](N)CO)CS/C=C/NC(=O)CNC1=O,SSNGC
P2,C[C@@H]1NC(=O)[C@@H]2CCCN2C(=O)[C@H](CO)NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC1=O,FSPAG"""

HELP_SEGS = [
    ("title", "环肽双通道预测 · 使用教程"),
    ("gap", ""),
    ("body", "输入环肽结构，同时给出两个通道的预测："),
    ("item", "· 定量：预测 MIC（µM）+ 90% 预测区间 + 信赖等级"),
    ("item", "· 定性：5 类活性概率 + 综合判定（阳性 / 阴性）"),
    ("gap", ""),

    ("h1", "一、预测（最常用）"),
    ("body", "1. 粘贴或导入含 SMILES 的表格"),
    ("body", "2. 填写测定条件（菌株 / 接种规格 / 培养基 / 环化类型），都可不填"),
    ("body", "3. 点「开始预测」"),
    ("gap", ""),
    ("warn", "⚠ 影响最大的是「环化类型」：不填会按「训练集未见过」处理，"),
    ("warn", "   预测向训练均值收缩（×0.51），数值偏保守。能填就填。"),
    ("gap", ""),

    ("h1", "二、诊断偏差 —— 什么时候做？"),
    ("body", "触发条件：手上已有一批这条肽的实测 MIC，想先确认模型在你的体系里准不准。"),
    ("gap", ""),
    ("body", "做法："),
    ("item", "1. 准备 CSV：SMILES, Strain, MIC_uM（列名大小写不敏感）"),
    ("item", "2. 到「诊断偏差」上传，点「开始诊断」"),
    ("item", "3. 看输出的「偏差（实测 − 预测）」："),
    ("ok",   "     |平均偏差| < 0.3 log  →  模型基本可用，不必重训"),
    ("bad",  "     |平均偏差| ≥ 0.3 log  →  存在系统性偏差，建议做标注学习"),
    ("gap", ""),

    ("h1", "三、标注学习 —— 什么时候做？"),
    ("body", "触发条件（满足任一）："),
    ("item", "· 你的肽属于训练集里没有的成环化学类型"),
    ("item", "· 诊断显示存在系统性偏差（|偏差| ≥ 0.3 log）"),
    ("item", "· 有一批同系列新肽的实测数据，希望模型针对它们更准"),
    ("gap", ""),
    ("body", "为什么必须重训：遇到没见过的成环类型时，零标注的 R² 只有约 0.02（不可用）。"),
    ("body", "只把预测整体平移的捷径我们验证过，仅能恢复 19–42% 的收益，所以必须重训。"),
    ("gap", ""),
    ("body", "需要多少标注（实测）："),
    ("tbl",  "       新增标注数          该类型 R²"),
    ("tbl",  "       ────────────────────────────"),
    ("tbl",  "            0              0.02  （不可用）"),
    ("hi",   "           10              0.147"),
    ("tbl",  "           25              0.248"),
    ("tbl",  "           50              0.334"),
    ("gap", ""),
    ("body", "做法："),
    ("item", "1. 准备 CSV：SMILES, Strain, MIC_uM[, CyclizationType]"),
    ("item", "2. 到「标注学习」上传，点「开始学习」"),
    ("item", "3. 后台重训约 3–8 分钟，完成后自动切换为新模型"),
    ("item", "4. 之后所有预测都用新模型（右上角显示当前模型）"),
    ("gap", ""),

    ("h1", "四、推荐的先后顺序"),
    ("flow", "预测  →  诊断偏差  →  标注学习  →  再预测"),
    ("body", "括号内为条件：预测（随时）→ 诊断（有实测值时）→ 标注学习（偏差大或遇新类型时）"),
    ("gap", ""),

    ("h1", "五、怎么读「判定」"),
    ("body", "规则：每个目标用「校准阳性率 ≥ 0.5」判阳；且"),
    ("item", "总抗菌 = 抗细菌 ∪ 革兰阳性 ∪ 革兰阴性（生物学约束）"),
    ("body", "输出示例：阳性：总抗菌、革兰阳性、革兰阴性    或    阴性"),
    ("gap", ""),

    ("h1", "六、怎么读「信赖等级」"),
    ("body", "四根轴：菌株在训练集内 / 环化类型在训练集内 / 结构相似度 ≥ 0.7 / 条件是否标准"),
    ("item", "全满足 → 高；违反 1 项 → 中；违反 ≥2 项 → 低；未指定菌株 → 仅定性"),
]

HELP_STYLE = {
    "title": ("Times New Roman", 16, "bold", "#1B3A57"),
    "h1":    ("Times New Roman", 12, "bold", "#2E6EB2"),
    "body":  ("Times New Roman", 11, "normal", "#33475B"),
    "item":  ("Times New Roman", 11, "normal", "#4A6076"),
    "warn":  ("Times New Roman", 11, "normal", "#C0622A"),
    "ok":    ("Times New Roman", 11, "normal", "#2E8B62"),
    "bad":   ("Times New Roman", 11, "bold", "#C0392B"),
    "tbl":   ("Times New Roman", 11, "normal", "#2E8B62"),
    "hi":    ("Times New Roman", 12, "bold", "#E2802C"),
    "flow":  ("Times New Roman", 13, "bold", "#2E8B62"),
    "gap":   ("Times New Roman", 6, "normal", "#000000"),
}

def _runs(s):
    """把字符串按 ASCII / 非 ASCII 切段，供中英分别选字体。"""
    out, cur, cur_lat = [], "", None
    for ch in s:
        lat = ord(ch) < 128
        if cur_lat is None or lat == cur_lat:
            cur += ch
            cur_lat = lat
        else:
            out.append((cur_lat, cur))
            cur, cur_lat = ch, lat
    if cur:
        out.append((cur_lat, cur))
    return out


FONT_LAT = "Times New Roman"
FONT_CJK = "宋体"



class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        root.title("环肽双通道预测")
        root.geometry("1240x895")
        root.resizable(False, False)
        root.configure(bg=BG)
        try:
            root.iconphoto(True, tk.PhotoImage(
                file=str(ASSETS / "logo.png")))
        except Exception:
            pass

        self._imgs = {}
        self.state = {"model": MODELS / "final_bundle.pkl", "rows": None,
                      "result": None, "busy": False}

        # 菌株 / 环化类型清单
        self.strains, self.cyc = [], []
        self._strain_labels = {}
        try:
            self.strains = pd.read_csv(DATA / "strains.csv")["strain"] \
                .astype(str).tolist()
        except Exception:
            pass
        # 带支撑度的显示标签（选择时再还原为原始菌株名）
        try:
            import json as _j
            sup = _j.loads((ASSETS / "strain_support.json")
                           .read_text(encoding="utf-8"))
        except Exception:
            sup = {}
        self._strain_raw = {}
        for st in self.strains:
            inf = sup.get(st)
            if inf:
                lab = f"{st}   ·  {inf['n']}对·{inf['level']}"
            else:
                lab = f"{st}   ·  未审计"
            self._strain_labels[lab] = st
            self._strain_raw[st] = lab
        self._strain_opts = list(self._strain_labels.keys())
        try:
            self.cyc = pd.read_csv(DATA / "cyc_types.csv")["type"] \
                .astype(str).tolist()
        except Exception:
            self.cyc = ["DSB", "AMD", "EST", "DCB", "TIE", "none"]

        self._topbar()
        self.body = tk.Frame(root, bg=BG)
        self.body.pack(fill="both", expand=True)
        self.pages = {}
        self._page_home()
        self._page_predict()
        self._page_learn()
        self._page_diag()
        self.show("home")

    # ══════════════════ 顶栏 ══════════════════
    def _topbar(self):
        tk, ttk = self.tk, self.ttk
        bar = tk.Frame(self.root, bg=CARD, height=86)
        self.topbar = bar
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=LINE, height=1).pack(fill="x", side="bottom")

        left = tk.Frame(bar, bg=CARD)
        left.pack(side="left", padx=18)
        try:
            img = tk.PhotoImage(file=str(ASSETS / "logo.png"))
            img = img.subsample(3, 3)          # 192 → 64
            self._imgs["logo"] = img
            tk.Label(left, image=img, bg=CARD).pack(side="left", pady=10)
        except Exception:
            pass
        tx = tk.Frame(left, bg=CARD)
        tx.pack(side="left", padx=12)
        tk.Label(tx, text="环肽双通道预测", bg=CARD, fg=INK,
                 font=("微软雅黑", 18, "bold")).pack(anchor="w")
        tk.Label(tx, text="定量 MIC + 定性活性 · 可标注学习",
                 bg=CARD, fg=INK2, font=(FONT_CJK, 10)).pack(anchor="w")

        right = tk.Frame(bar, bg=CARD)
        right.pack(side="right", padx=18)
        self.model_lbl_top = tk.Label(
            right, text=f"当前模型：{self.state['model'].name}",
            bg=CARD, fg=INK2, font=(FONT_CJK, 10))
        self.model_lbl_top.pack(side="right", padx=(12, 0))
        ttk.Button(right, text="?  帮助", width=10,
                   command=self._help).pack(side="right")

    def _help(self):
        tk = self.tk
        w = tk.Toplevel(self.root)
        w.title("使用教程")
        w.geometry("840x820")
        w.configure(bg=CARD)
        holder = tk.Frame(w, bg=CARD)
        holder.pack(fill="both", expand=True)
        t = tk.Text(holder, wrap="word", bg=CARD, fg=INK, bd=0, padx=22,
                    pady=16, spacing1=3, spacing3=3)
        vsb = tk.Scrollbar(holder, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        t.pack(side="left", fill="both", expand=True)

        # 每个「一、二、三…」分栏用不同颜色
        H1_COLORS = ["#2E6EB2", "#2E8B62", "#D9772B", "#7A5BA6",
                     "#B8860B", "#3E7C8C"]

        # 每种样式配「拉丁 / 中文」两个标签
        for tag, (fam, size, weight, color) in HELP_STYLE.items():
            for suf, f2 in (("lat", FONT_LAT), ("cjk", FONT_CJK)):
                t.tag_configure(
                    f"{tag}_{suf}", font=(f2, size, weight), foreground=color,
                    spacing1=8 if (tag == "h1" and suf == "lat") else 2,
                    spacing3=5 if (tag == "h1" and suf == "lat") else 1)

        h1i = 0
        for tag, txt in HELP_SEGS:
            tg = tag
            if tag == "h1":
                tg = f"h1x{h1i}"
                col = H1_COLORS[h1i % len(H1_COLORS)]
                for suf, f2 in (("lat", FONT_LAT), ("cjk", FONT_CJK)):
                    t.tag_configure(f"{tg}_{suf}", font=(f2, 12, "bold"),
                                    foreground=col, spacing1=9,
                                    spacing3=6)
                h1i += 1
            if not txt:
                t.insert("end", "\n", f"{tg}_lat")
                continue
            for is_lat, seg in _runs(txt):
                t.insert("end", seg,
                         f"{tg}_{'lat' if is_lat else 'cjk'}")
            t.insert("end", "\n", f"{tg}_lat")
        t.config(state="disabled")
        tk.Button(w, text="关闭", command=w.destroy, bg=HI, fg="white",
                  relief="flat", font=("微软雅黑", 10),
                  width=10).pack(pady=10)

    # ══════════════════ 页面切换 ══════════════════
    def show(self, name):
        for f in self.pages.values():
            f.pack_forget()
        if (getattr(self, "topbar", None) is not None
                and not self.topbar.winfo_ismapped()):
            self.topbar.pack(fill="x", side="top", before=self.body)
        self.pages[name].pack(fill="both", expand=True)

    def _card(self, parent, title=None):
        tk = self.tk
        outer = tk.Frame(parent, bg=BG)
        f = tk.Frame(outer, bg=CARD, highlightbackground=LINE,
                     highlightthickness=1)
        f.pack(fill="both", expand=True)
        if title:
            tk.Label(f, text=title, bg=CARD, fg=INK,
                     font=("微软雅黑", 10, "bold")).pack(
                anchor="w", padx=14, pady=(9, 3))
        return outer, f

    # ══════════════════ 首页 ══════════════════
    CAT_FONT = "华文琥珀"

    CAT_FONT = "华文琥珀"

    # 首页版式（对齐参考图）
    CARD_W, CARD_H = 364, 356
    CARD_GAP = 34
    CARD_TOP = 318

    def _page_home(self):
        tk = self.tk
        p = tk.Frame(self.body, bg=BG)
        self.pages["home"] = p

        import json as _json
        L = {}
        try:
            L = _json.loads((ASSETS / "art_layout.json")
                            .read_text(encoding="utf-8"))
        except Exception:
            pass
        self.home_layout = L
        iw, ih = L.get("img", [1240, 809])

        cv = tk.Canvas(p, width=iw, height=ih, bd=0, highlightthickness=0,
                       bg="#F4F9FE")
        cv.pack(fill="both", expand=True)
        self.home_cv = cv

        try:
            bg = tk.PhotoImage(file=str(ASSETS / "art_bg.png"))
            self._imgs["art_bg"] = bg
            cv.create_image(0, 0, image=bg, anchor="nw")
        except Exception:
            pass

        # 标题（原生）
        cv.create_text(iw // 2, L.get("title_y", 75), text="选择功能",
                       fill="#17395B", font=("华文琥珀", 32))
        cv.create_text(iw // 2, L.get("sub_y", 140),
                       text="定量 MIC  ·  定性活性  ·  可标注学习",
                       fill="#3E6B96", font=("微软雅黑", 14))

        specs = [("predict", "预  测", "Prediction",
                  "输入结构 → 双通道结果\n含 90% 区间与信赖等级"),
                 ("learn", "标注学习", "Fine-tuning",
                  "用自有标注重训\n让模型针对你的肽更准"),
                 ("diag", "诊断偏差", "Diagnostics",
                  "先看模型准不准\n再决定要不要重训")]
        zones = {}
        for key, cn, en, desc in specs:
            d = L.get(key)
            if not d:
                continue
            x, y, w, h = d["x"], d["y"], d["w"], d["h"]
            zones[key] = (x, y, x + w, y + h)
            try:
                im = tk.PhotoImage(file=str(ASSETS / f"art_card_{key}.png"))
                im = im.subsample(2, 2)      # 790×780 → 395×390（含 20px 投影留白）
                self._imgs[f"art_{key}"] = im
                cv.create_image(x - 20, y - 20, image=im, anchor="nw",
                                tags=(f"c{key}",))
            except Exception:
                cv.create_rectangle(x, y, x + w, y + h, fill="#2E6EB2",
                                    outline="", tags=(f"c{key}",))
            cx = x + w / 2
            # 文字位置按设计图比例（相对卡片高度）
            cv.create_text(cx, y + h * 0.570, text=cn, fill="#FFFFFF",
                           font=("微软雅黑", 21, "bold"), tags=(f"c{key}",))
            cv.create_text(cx, y + h * 0.678, text=en, fill="#F2F8FF",
                           font=("Times New Roman", 15, "italic"),
                           tags=(f"c{key}",))
            cv.create_text(cx, y + h * 0.838, text=desc, fill="#F6FBFF",
                           justify="center", font=("微软雅黑", 10),
                           anchor="center", tags=(f"c{key}",))

        self.home_zones = zones

        def _hit(ev):
            for key, (x1, y1, x2, y2) in zones.items():
                if x1 <= ev.x <= x2 and y1 <= ev.y <= y2:
                    self.show(key)
                    return

        def _cur(ev):
            on = any(v[0] <= ev.x <= v[2] and v[1] <= ev.y <= v[3]
                     for v in zones.values())
            cv.configure(cursor="hand2" if on else "")

        cv.bind("<Button-1>", _hit)
        cv.bind("<Motion>", _cur)

    @staticmethod
    def _card_color(key):
        return {"predict": "#1F4E79", "learn": "#9C5A1E",
                "diag": "#1F5C43"}.get(key, CARD)

    # ══════════════════ 通用：返回 / 清空 ══════════════════
    def _back_row(self, parent, clear_cmd):
        tk, ttk = self.tk, self.ttk
        r = tk.Frame(parent, bg=BG)
        r.pack(fill="x", padx=16, pady=(10, 4))
        ttk.Button(r, text="← 返回", width=9,
                   command=lambda: self.show("home")).pack(side="left")
        ttk.Button(r, text="清空", width=8,
                   command=clear_cmd).pack(side="right")

    # ══════════════════ 预测页 ══════════════════
    def _page_predict(self):
        tk, ttk = self.tk, self.ttk
        P = tk.Frame(self.body, bg=BG)
        self.pages["predict"] = P
        self._back_row(P, lambda: self._clear_predict())

        # ① 样例（独立卡片，不可选中/编辑）
        wrap, card = self._card(P, "① 输入格式示例（只读，不可编辑）")
        wrap.pack(fill="x", padx=16, pady=4)
        samp = tk.Text(card, height=3, wrap="none", font=("Consolas", 9),
                       bg="#F7FAFE", fg=INK2, bd=0, padx=10, pady=8,
                       cursor="arrow", takefocus=0)
        samp.pack(fill="x", padx=10, pady=(0, 10))
        samp.insert("1.0", SAMPLE)
        samp.config(state="disabled")
        # 禁用选中
        samp.bind("<Button-1>", lambda e: "break")
        samp.bind("<B1-Motion>", lambda e: "break")
        samp.bind("<Control-c>", lambda e: "break")

        # ② 输入
        wrap2, card2 = self._card(P, "② 输入数据")
        wrap2.pack(fill="x", padx=16, pady=4)
        tb = tk.Frame(card2, bg=CARD)
        tb.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(tb, text="导入文件…",
                   command=lambda: self._import(txt)).pack(side="left")
        tk.Label(tb, text="或直接粘贴到下方（每行 ID,SMILES[,序列]）",
                  bg=CARD, fg=INK2,
                  font=(FONT_CJK, 10)).pack(side="left", padx=8)
        txt = tk.Text(card2, height=3, wrap="none", font=("Consolas", 9),
                      bg="#FBFDFF", fg=INK, bd=1, relief="solid")
        txt.pack(fill="x", padx=10, pady=(0, 10))

        # ③ 测定条件
        wrap3, card3 = self._card(
            P, "③ 测定条件（可选；填得越全，区间越窄、等级越高）")
        wrap3.pack(fill="x", padx=16, pady=4)
        g = tk.Frame(card3, bg=CARD)
        g.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(g, text="菌株", bg=CARD, fg=INK,
                 font=(FONT_CJK, 10)).grid(row=0, column=0,
                                                   sticky="w", pady=4)
        _def = getattr(self, "_strain_raw", {}).get(
            "Escherichia coli K-12", "Escherichia coli K-12")
        self.strain_var = tk.StringVar(value=_def)
        cb = ttk.Combobox(g, textvariable=self.strain_var, width=52,
                          values=self._strain_opts)
        cb.grid(row=0, column=1, sticky="w", padx=(6, 6))
        tk.Label(g, text="输入前几个字母可筛选", bg=CARD, fg=INK2,
                 font=(FONT_CJK, 9)).grid(row=0, column=2, sticky="w")

        def _filter(_e=None):
            typed = self.strain_var.get().strip().lower()
            hit = [x for x in self._strain_opts
                   if x.lower().startswith(typed)]
            cb["values"] = hit if (typed and hit) else self._strain_opts
        cb.bind("<KeyRelease>", _filter)

        tk.Label(g, text="接种规格", bg=CARD, fg=INK,
                 font=(FONT_CJK, 10)).grid(row=1, column=0,
                                                   sticky="w", pady=4)
        self.cfu_var = tk.StringVar(value=CFU_OPTS[3])
        ttk.Combobox(g, textvariable=self.cfu_var, width=22, state="readonly",
                     values=CFU_OPTS).grid(row=1, column=1, sticky="w",
                                           padx=(6, 6))

        tk.Label(g, text="培养基", bg=CARD, fg=INK,
                 font=(FONT_CJK, 10)).grid(row=2, column=0,
                                                   sticky="w", pady=4)
        self.med_var = tk.StringVar(value="MHB")
        ttk.Combobox(g, textvariable=self.med_var, width=22, state="readonly",
                     values=MED_OPTS).grid(row=2, column=1, sticky="w",
                                           padx=(6, 6))

        tk.Label(g, text="环化类型", bg=CARD, fg=INK,
                 font=(FONT_CJK, 10)).grid(row=3, column=0,
                                                   sticky="w", pady=4)
        self.type_var = tk.StringVar(value="（未提供）")
        ttk.Combobox(g, textvariable=self.type_var, width=22, state="readonly",
                     values=["（未提供）"] + self.cyc).grid(
            row=3, column=1, sticky="w", padx=(6, 6))
        tk.Label(g, text="⚠ 不填则按「新类型」处理，预测偏保守", bg=CARD,
                 fg="#C0622A", font=(FONT_CJK, 9)).grid(
            row=3, column=2, sticky="w")

        # ④ 执行
        bar = tk.Frame(P, bg=BG)
        bar.pack(fill="x", padx=16, pady=(6, 2))
        self.run_btn = ttk.Button(bar, text="开始预测", command=self._run)
        self.run_btn.pack(side="left")
        self.info = tk.Label(bar, text="就绪", bg=BG, fg=INK2,
                             font=(FONT_CJK, 10))
        self.info.pack(side="left", padx=12)

        # 明细面板（底部固定高度 + 滚动条，先占位再让结果卡扩张）
        dt = tk.Frame(P, bg=BG, height=118)
        dt.pack(side="bottom", fill="x", padx=16, pady=(0, 10))
        dt.pack_propagate(False)
        self.dtxt = tk.Text(dt, wrap="word", font=("Consolas", 9),
                            bg="#FBFDFF", fg=INK, bd=1, relief="solid",
                            padx=6, pady=4)
        dsb = tk.Scrollbar(dt, orient="vertical", command=self.dtxt.yview)
        self.dtxt.configure(yscrollcommand=dsb.set)
        dsb.pack(side="right", fill="y")
        self.dtxt.pack(side="left", fill="both", expand=True)
        self.dtxt.insert("1.0", "（点选上方一行查看两个通道的完整依据）")
        self.dtxt.config(state="disabled")

        # 结果
        wrap4, card4 = self._card(P, "④ 结果")
        wrap4.pack(fill="both", expand=True, padx=16, pady=(2, 10))
        cols = ["ID", "定量_uM", "定量区间", "信赖等级", "定性_总抗菌",
                "定性_抗细菌", "定性_抗革兰阳性", "定性_抗革兰阴性",
                "定性_抗真菌", "判定"]
        W = {"ID": 50, "定量_uM": 70, "定量区间": 100, "信赖等级": 55,
             "定性_总抗菌": 72, "定性_抗细菌": 72, "定性_抗革兰阳性": 82,
             "定性_抗革兰阴性": 82, "定性_抗真菌": 68, "判定": 250}
        tree = ttk.Treeview(card4, columns=cols, show="headings", height=3)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=W.get(c, 80), anchor="w")
        tree.pack(fill="both", expand=True, side="left", padx=(10, 0),
                  pady=(0, 8))
        sb = ttk.Scrollbar(card4, orient="vertical", command=tree.yview)
        sb.pack(side="right", fill="y", padx=(0, 10), pady=(0, 8))
        tree.configure(yscrollcommand=sb.set)
        self.tree, self.cols = tree, cols
        tree.bind("<<TreeviewSelect>>", self._detail)

        self.p_txt = txt

    def _clear_predict(self):
        self.p_txt.delete("1.0", "end")
        self.strain_var.set(getattr(self, "_strain_raw", {}).get(
            "Escherichia coli K-12", "Escherichia coli K-12"))
        self.cfu_var.set(CFU_OPTS[3])
        self.med_var.set("MHB")
        self.type_var.set("（未提供）")
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.state["rows"] = None
        self.state["result"] = None
        self.dtxt.config(state="normal")
        self.dtxt.delete("1.0", "end")
        self.dtxt.insert("1.0", "（点选上方一行查看两个通道的完整依据）")
        self.dtxt.config(state="disabled")
        self.info.config(text="已清空")

    def _import(self, widget):
        from tkinter import filedialog, messagebox
        import cpamp_tool as T
        f = filedialog.askopenfilename(
            filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls"), ("全部", "*.*")])
        if not f:
            return
        try:
            df = T._read_table(f)
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return
        cs = T._col(df, ["smiles", "结构", "分子"])
        ci = T._col(df, ["id", "序号", "编号", "name"])
        if cs is None:
            messagebox.showerror("缺少 SMILES 列",
                                 f"检测到：{list(df.columns)}")
            return
        lines = []
        for i, r in df.iterrows():
            rid = str(r[ci]) if ci else f"P{i+1}"
            lines.append(f"{rid},{r[cs]}")
        widget.delete("1.0", "end")
        widget.insert("1.0", "\n".join(lines))
        self.info.config(text=f"已载入 {len(lines)} 条")

    def _parse(self):
        rows = []
        for ln in self.p_txt.get("1.0", "end").strip().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = [x.strip() for x in
                 (ln.split("\t") if "\t" in ln else ln.split(","))]
            if len(p) == 1:
                rows.append({"ID": f"P{len(rows)+1}", "SMILES": p[0]})
            elif len(p) == 2:
                rows.append({"ID": p[0], "SMILES": p[1]})
            else:
                rows.append({"ID": p[0], "SMILES": p[1], "Seq": p[2]})
        return pd.DataFrame(rows)

    def _run(self):
        from tkinter import messagebox
        if self.state["busy"]:
            return
        df = self._parse()
        if df.empty:
            messagebox.showwarning("提示", "没有可解析的数据")
            return
        d = df.copy()
        _sv = self.strain_var.get().strip()
        d["Strain"] = self._strain_labels.get(_sv, _sv)
        c = CFU_VAL.get(self.cfu_var.get())
        if c is not None:
            d["cfuGroup"] = c
        if self.med_var.get() != MED_OPTS[0]:
            d["Medium"] = self.med_var.get()
        if self.type_var.get() != "（未提供）":
            d["CyclizationType"] = self.type_var.get()
        self.run_btn.config(state="disabled")
        self.info.config(text=f"预测中…（{len(d)} 条）")
        self.state["busy"] = True

        def work():
            try:
                from cpamp_predict_v2 import predict, to_compact
                r = predict(d, use_esm=True, calibrate=True, verbose=False,
                            bundle=self.state["model"])
                c2 = to_compact(r)
                self.state["result"] = r
                self.state["rows"] = r
                self.root.after(0, lambda: self._fill(c2))
            except Exception:
                tb = traceback.format_exc()[-700:]
                self.root.after(0, lambda: messagebox.showerror(
                    "预测失败", tb))
            finally:
                self.state["busy"] = False
                self.root.after(0, lambda: (
                    self.run_btn.config(state="normal"),
                    self.info.config(text="完成")))
        threading.Thread(target=work, daemon=True).start()

    def _fill(self, c2):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for i, (_, row) in enumerate(c2.iterrows()):
            vals = []
            for c in self.cols:
                v = row.get(c, "")
                if isinstance(v, float) and pd.notna(v):
                    v = f"{v:.3g}"
                vals.append("" if (isinstance(v, float) and pd.isna(v)) else v)
            self.tree.insert("", "end", iid=str(i), values=vals)
        self.info.config(text=f"完成 {len(c2)} 条")

    def _detail(self, _e=None):
        sel = self.tree.selection()
        if not sel or self.state["rows"] is None:
            return
        r = self.state["rows"].iloc[int(sel[0])]
        L = ["【定量通道】",
             f"  MIC = {r.get('定量_uM', float('nan')):.3g} µM"
             f"   90% 区间 [{r.get('定量_uM_low90', float('nan')):.3g},"
             f" {r.get('定量_uM_high90', float('nan')):.3g}]"
             f"   σ = {r.get('σ_log10', float('nan')):.3f}",
             f"  信赖等级：{r.get('信赖等级','')}  依据：{r.get('说明','')}", "",
             "【定性通道】目标 / 概率 / 校准阳性率（Wilson 95%）/ 可靠度"]
        for nm in ("总抗菌(H2)", "抗细菌", "抗革兰阳性", "抗革兰阴性", "抗真菌"):
            p_ = r.get(f"定性_{nm}", None)
            if p_ is None or (isinstance(p_, float) and pd.isna(p_)):
                continue
            L.append(f"  {nm:10s} p={p_:.3f}   "
                     f"{r.get(f'定性_{nm}_校准','')}   "
                     f"{r.get(f'定性_{nm}_可靠度','')}")
        L += ["", f"  判定：{r.get('判定','')}", "",
              f"  口径：{r.get('区间说明','')}"]
        self.dtxt.config(state="normal")
        self.dtxt.delete("1.0", "end")
        self.dtxt.insert("1.0", "\n".join(L))
        self.dtxt.config(state="disabled")

    # ══════════════════ 标注学习页 ══════════════════
    def _page_learn(self):
        tk, ttk = self.tk, self.ttk
        P = tk.Frame(self.body, bg=BG)
        self.pages["learn"] = P
        self._back_row(P, lambda: self._clear_learn())

        wrap, card = self._card(P, "什么时候需要标注学习")
        wrap.pack(fill="x", padx=16, pady=4)
        body = tk.Frame(card, bg=CARD)
        body.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(body, justify="left", bg=CARD, fg=INK,
                 font=(FONT_CJK, 10),
                 text="遇到训练集里没有的成环化学类型时，零标注的 R² 只有约 0.02，不可用；\n"
                      "用少量自有标注重训后即可使用。实测收益见下表。"
                 ).pack(anchor="w")

        mid = tk.Frame(body, bg=CARD)
        mid.pack(anchor="w", pady=6)
        tab = ttk.Treeview(mid, columns=("a", "b"), show="headings",
                           height=5, selectmode="none")
        tab.heading("a", text="新增标注数")
        tab.heading("b", text="该类型 R²")
        tab.column("a", width=150, anchor="center")
        tab.column("b", width=150, anchor="center")
        tab.pack()
        for i, (n, r) in enumerate((("0", "0.02（不可用）"), ("10", "0.147"),
                                    ("25", "0.248"), ("50", "0.334"))):
            tab.insert("", "end", values=(n, r), tags=("hl",) if i == 1 else ())
        tab.tag_configure("hl", background="#FFF4E6")

        tk.Label(body, justify="left", bg=CARD, fg="#C0622A",
                 font=(FONT_CJK, 9),
                 text="注：只把预测整体平移（不重训）仅能恢复 19–42% 的收益，故必须重训。"
                 ).pack(anchor="w")

        wrap2, card2 = self._card(
            P, "上传标注（CSV / Excel：SMILES, Strain, MIC_uM[，CyclizationType]）")
        wrap2.pack(fill="x", padx=16, pady=4)
        row = tk.Frame(card2, bg=CARD)
        row.pack(fill="x", padx=10, pady=(0, 10))
        self.learn_path = tk.StringVar(value="（未选择）")
        tk.Entry(row, textvariable=self.learn_path, width=74,
                 bg="#FBFDFF", fg=INK, bd=1, relief="solid").pack(
            side="left", padx=(0, 6))
        ttk.Button(row, text="浏览…", command=self._pick_learn).pack(
            side="left")

        bar = tk.Frame(P, bg=BG)
        bar.pack(fill="x", padx=16, pady=6)
        self.learn_btn = ttk.Button(bar, text="开始学习", command=self._learn)
        self.learn_btn.pack(side="left")
        self.learn_info = tk.Label(bar, text="", bg=BG, fg=INK2,
                                   font=(FONT_CJK, 10))
        self.learn_info.pack(side="left", padx=12)

        wrap3, card3 = self._card(P, "日志")
        wrap3.pack(fill="both", expand=True, padx=16, pady=(2, 10))
        self.learn_log = tk.Text(card3, height=9, wrap="word",
                                 font=("Consolas", 9), bg="#FBFDFF", fg=INK,
                                 bd=0, padx=8, pady=8)
        self.learn_log.pack(fill="both", expand=True)

    def _clear_learn(self):
        self.learn_path.set("（未选择）")
        self.learn_log.delete("1.0", "end")
        self.learn_info.config(text="已清空")

    def _pick_learn(self):
        from tkinter import filedialog
        f = filedialog.askopenfilename(
            filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls"), ("全部", "*.*")])
        if f:
            self.learn_path.set(f)

    def _learn(self):
        from tkinter import messagebox
        import cpamp_tool as T
        f = self.learn_path.get()
        if not f or f == "（未选择）":
            messagebox.showwarning("提示", "请先选择标注文件")
            return
        self.learn_btn.config(state="disabled")
        self.learn_log.delete("1.0", "end")
        self.learn_info.config(text="重训中，请勿关闭窗口…")

        def log(m):
            self.learn_log.insert("end", m + "\n")
            self.learn_log.see("end")

        def work():
            t0 = time.time()
            try:
                lab = T.load_labels(f)
                self.root.after(0, lambda: log(f"读入标注 {len(lab)} 条"))
                out = MODELS / "finetuned_bundle.pkl"
                T.cmd_finetune(type("A", (), {
                    "labels": f, "out": str(out), "base": None})())
                dt = time.time() - t0
                self.state["model"] = out
                self.root.after(0, lambda: (
                    log(f"完成，用时 {dt:.0f} 秒"),
                    log(f"新模型：{out.name}"),
                    self.model_lbl.config(text=f"当前模型：{out.name}"),
                    self.model_lbl_top.config(
                        text=f"当前模型：{out.name}"),
                    self.learn_info.config(text="已切换到新模型")))
            except Exception:
                tb = traceback.format_exc()[-1000:]
                self.root.after(0, lambda: (
                    log("失败：\n" + tb),
                    self.learn_info.config(text="失败")))
            finally:
                self.root.after(0, lambda: self.learn_btn.config(
                    state="normal"))
        threading.Thread(target=work, daemon=True).start()

    # ══════════════════ 诊断页 ══════════════════
    def _page_diag(self):
        tk, ttk = self.tk, self.ttk
        P = tk.Frame(self.body, bg=BG)
        self.pages["diag"] = P
        self._back_row(P, lambda: self._clear_diag())

        wrap, card = self._card(P, "① 数据格式要求（先按此格式处理数据，再上传）")
        wrap.pack(fill="x", padx=16, pady=4)
        tk.Label(card, justify="left", bg=CARD, fg=INK,
                 font=("Consolas", 9),
                 text="SMILES        分子结构（也认「结构」「分子」）\n"
                      "Strain        菌株名，需与训练集一致（也认「菌株」）\n"
                      "MIC_uM        实测 MIC，单位 µM（也认「MIC」）\n"
                      "CyclizationType  建议提供（环化类型），便于定位偏差来源"
                 ).pack(anchor="w", padx=12, pady=(0, 10))

        wrap2, card2 = self._card(P, "② 上传实测数据")
        wrap2.pack(fill="x", padx=16, pady=4)
        row = tk.Frame(card2, bg=CARD)
        row.pack(fill="x", padx=10, pady=(0, 10))
        self.diag_path = tk.StringVar(value="（未选择）")
        tk.Entry(row, textvariable=self.diag_path, width=74,
                 bg="#FBFDFF", fg=INK, bd=1, relief="solid").pack(
            side="left", padx=(0, 6))
        ttk.Button(row, text="浏览…", command=self._pick_diag).pack(
            side="left")

        tk.Label(P, justify="left", bg=BG, fg=INK2,
                 font=(FONT_CJK, 10),
                 text="③ 推荐顺序：  处理数据 → 诊断偏差 → （偏差大时）标注学习 → 再预测"
                 ).pack(anchor="w", padx=18, pady=(8, 2))

        bar = tk.Frame(P, bg=BG)
        bar.pack(fill="x", padx=16, pady=4)
        ttk.Button(bar, text="开始诊断", command=self._diag).pack(side="left")
        self.diag_info = tk.Label(bar, text="", bg=BG, fg=INK2,
                                  font=(FONT_CJK, 10))
        self.diag_info.pack(side="left", padx=12)

        wrap3, card3 = self._card(P, "④ 诊断结果")
        wrap3.pack(fill="both", expand=True, padx=16, pady=(2, 10))
        self.diag_out = tk.Text(card3, height=9, wrap="word",
                                font=("Consolas", 9), bg="#FBFDFF", fg=INK,
                                bd=0, padx=10, pady=8)
        self.diag_out.pack(fill="both", expand=True)

    def _clear_diag(self):
        self.diag_path.set("（未选择）")
        self.diag_out.delete("1.0", "end")
        self.diag_info.config(text="已清空")

    def _pick_diag(self):
        from tkinter import filedialog
        f = filedialog.askopenfilename(
            filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls"), ("全部", "*.*")])
        if f:
            self.diag_path.set(f)

    def _diag(self):
        from tkinter import messagebox
        import numpy as np
        import cpamp_tool as T
        f = self.diag_path.get()
        if not f or f == "（未选择）":
            messagebox.showwarning("提示", "请先选择文件")
            return
        self.diag_info.config(text="计算中…")

        def work():
            try:
                from cpamp_predict_v2 import predict
                lab = T.load_labels(f)
                r = predict(lab, use_esm=True, calibrate=True, verbose=False,
                            bundle=self.state["model"])
                pred = pd.to_numeric(r.get("定量_log10uM"),
                                     errors="coerce").to_numpy(float)
                y = lab["y"].to_numpy(float)
                d = y - pred
                ok = np.isfinite(d)
                n = int(ok.sum())
                mu = float(d[ok].mean())
                txt = (f"可比较 {n} 条\n\n"
                       f"偏差（实测 − 预测）\n"
                       f"  均值 {mu:+.4f}   中位 {np.median(d[ok]):+.4f}"
                       f"   SD {d[ok].std():.4f}\n"
                       f"  平均绝对误差 {np.mean(np.abs(d[ok])):.4f} log\n\n")
                if abs(mu) >= 0.3:
                    txt += (f"⇒ 存在系统性偏差 {mu:+.2f} log"
                            f"（约 {10**mu:.1f} 倍）\n"
                            f"   建议：到「标注学习」用这份数据重训，再预测。")
                else:
                    txt += "⇒ 未见明显系统偏差（|偏差| < 0.3 log），可直接使用。"
                self.root.after(0, lambda: (
                    self.diag_out.delete("1.0", "end"),
                    self.diag_out.insert("1.0", txt),
                    self.diag_info.config(text="完成")))
            except Exception:
                tb = traceback.format_exc()[-800:]
                self.root.after(0, lambda: (
                    messagebox.showerror("诊断失败", tb),
                    self.diag_info.config(text="失败")))
        threading.Thread(target=work, daemon=True).start()


def _dep_check():
    """启动前检查关键依赖；缺失时给出可执行的修复命令。"""
    miss = []
    for mod, pkg in (("lightgbm", "lightgbm"), ("rdkit", "rdkit"),
                     ("numpy", "numpy"), ("pandas", "pandas"),
                     ("scipy", "scipy")):
        try:
            __import__(mod)
        except Exception:
            miss.append(pkg)
    if not miss:
        return True
    import tkinter as tk
    from tkinter import messagebox
    r = tk.Tk(); r.withdraw()
    msg = (
        "缺少依赖：" + "、".join(miss) + "\n\n"
        "当前解释器：\n  " + sys.executable + "\n\n"
        "请改用项目虚拟环境启动：\n"
        "  ..\\.venv\\Scripts\\python.exe cpamp_gui.py\n\n"
        "或直接双击项目里的「start.bat」。\n\n"
        "若虚拟环境也没装，执行：\n"
        "  ..\\.venv\\Scripts\\pip install " + " ".join(miss)
    )
    messagebox.showerror("依赖缺失", msg)
    r.destroy()
    return False

def main():
    import tkinter as tk
    if not _dep_check():
        return 2
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
