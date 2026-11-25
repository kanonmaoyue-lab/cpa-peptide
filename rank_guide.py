"""rank_guide.py -- Hit-to-Lead 优先级排序与富集指导。

═══ 为什么需要它 ═══
    论文的核心结论是：本系统在真实部署口径下的价值不在"绝对 MIC 预测"
    （前瞻时间切分 R² 仅 0.3201），而在"把高活性分子排到前面"
    （EF₁₀ = 2.42 / 4.00）。因此推理接口除了给点预测与区间，还应直接给出
    **合成优先级排序**，并把"这个排序有多可信"如实告知使用者。

═══ 依据（全部来自论文前瞻时间切分，n = 7 919）═══
    模型      EF₁₀ = 2.42（MIC ≤ 4 µM） / 4.00（MIC ≤ 1 µM）
    召回      前 10% 覆盖 24.2% / 39.9% 的真实高活性分子
    基线      按结构相似度排序 1.07；按菌株中位值 1.48
    上限      1/基础率 = 3.03 / 7.42

═══ 设计原则 ═══
    · 排序键用定量点预测（升序）；无定量值（未给菌株）时退化为定性概率（降序）
    · 明确标注"优先级 ≠ 绝对活性判定"，并给出召回率，避免过度承诺
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

# ── 论文口径的富集证据（前瞻时间切分，3 seed 集成）──
VALIDATION = {
    "n_test": 7919,
    "EF10_MIC4": 2.42,
    "EF10_MIC1": 4.00,
    "recall_MIC4": 0.242,
    "recall_MIC1": 0.399,
    "base_MIC4": 0.330,
    "base_MIC1": 0.135,
    "ceiling_MIC4": 3.03,
    "ceiling_MIC1": 7.42,
    "baseline_similarity": 1.07,
    "baseline_strain_median": 1.48,
}

QUANT_COL = "定量_uM"
QUAL_COL = "定性_总抗菌(H2)"
QUAL_COL_ALT = "定性_总抗菌"

TIER_TOP = "优先合成"
TIER_MID = "备选"
TIER_LOW = "低优先"
TIER_NQ = "无定量值"
TIER_NA = "无法排序"


def _qual_col(df: pd.DataFrame) -> Optional[str]:
    for c in (QUAL_COL, QUAL_COL_ALT):
        if c in df.columns:
            return c
    return None


def rank_by_activity(df: pd.DataFrame, top: float = 0.10,
                     mid: float = 0.50) -> tuple[pd.DataFrame, dict]:
    """按预测活性给出合成优先级。

    参数
    ----
    df  : predict() 的输出（完整列或精简列均可）
    top : 「优先合成」的比例，默认前 10%（与论文 EF₁₀ 口径一致）
    mid : 「备选」的上界，默认前 50%

    返回
    ----
    (排序后的 DataFrame, 汇总信息 dict)
      新增列：`优先级`（1..N）、`建议`（优先合成/备选/低优先/无定量值）
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    q = pd.to_numeric(out.get(QUANT_COL), errors="coerce") \
        if QUANT_COL in out.columns else pd.Series([float("nan")] * n)

    if q.notna().any():
        # 有定量：按预测 MIC 升序；缺失值排最后，同值按定性概率降序打破平局
        qc = _qual_col(out)
        tmp = out.assign(_q=q)
        by = ["_q"] + ([qc] if qc else [])
        asc = [True] + ([False] if qc else [])
        out = tmp.sort_values(by, ascending=asc, na_position="last",
                              kind="mergesort").drop(columns="_q").reset_index(drop=True)
        mode = "quant"
    else:
        qc = _qual_col(out)
        if qc:
            out = out.sort_values(qc, ascending=False, kind="mergesort").reset_index(drop=True)
            mode = "qual"
        else:
            mode = "none"

    # 缺定量与定性两列时无法排序：不分档，避免按输入行号误标「优先合成」
    if mode == "none":
        out.insert(0, "优先级", [None] * n)
        out.insert(1, "建议", [TIER_NA] * n)
        return out, {
            "n": n, "top_frac": top, "k_top": 0, "k_mid": 0,
            "mode": mode, "n_no_quant": n,
            "tiers": {TIER_NA: n},
            "note": "缺少定量与定性列，无法排序；不按行号分档",
        }

    # 分档
    k_top = max(1, int(math.ceil(n * top))) if n else 0
    k_mid = max(k_top, int(math.ceil(n * mid))) if n else 0
    tiers, ranks = [], []
    for i in range(n):
        tiers.append(TIER_TOP if i < k_top else
                     TIER_MID if i < k_mid else TIER_LOW)
        ranks.append(i + 1)
    # 无定量值的行（排序后位于尾部）单独标注
    if q.notna().any() and q.isna().any():
        nq = int(q.isna().sum())
        for i in range(n - nq, n):
            tiers[i] = TIER_NQ
    out.insert(0, "优先级", ranks)
    out.insert(1, "建议", tiers)

    summary = {
        "n": n, "top_frac": top, "k_top": k_top, "k_mid": k_mid,
        "mode": mode, "n_no_quant": int(q.isna().sum()) if n else 0,
        "tiers": {t: tiers.count(t) for t in
                  (TIER_TOP, TIER_MID, TIER_LOW, TIER_NQ)},
    }
    return out, summary


def to_ranked_compact(ranked: pd.DataFrame, compact_fn) -> pd.DataFrame:
    """把排序结果压成展示列，同时保留「优先级 / 建议」。

    `to_compact()` 只保留固定列名，会把排序新增的两列丢掉，故在此补回。
    传 `compact_fn`（通常是 `cpamp_predict_v2.to_compact`）以保持依赖单向。
    """
    c = compact_fn(ranked).reset_index(drop=True)
    if "定量_uM" in c.columns:
        # 排序表直接交给合成排期，保留 4 位有效数字即可
        c["定量_uM"] = pd.to_numeric(c["定量_uM"], errors="coerce").map(
            lambda v: v if pd.isna(v) else float(f"{v:.4g}"))
    if "优先级" in ranked.columns:
        c.insert(0, "优先级", ranked["优先级"].to_numpy())
    if "建议" in ranked.columns:
        c.insert(1, "建议", ranked["建议"].to_numpy())
    return c


def guidance_text(summary: dict, compact: bool = False) -> str:
    """把论文证据转成给使用者看的一段说明（CLI / GUI 共用）。"""
    if summary.get("mode") == "none":
        return ("这批结果既无定量 MIC 值、也无定性概率，无法给出优先级排序。"
                "请指定菌株（得到定量通道输出），或确认定性模型可用后重试。")
    v = VALIDATION
    k = summary.get("k_top", 0)
    n = summary.get("n", 0)
    head = (f"已按预测活性排序：{n} 条中前 {summary.get('top_frac', 0.1)*100:.0f}%"
            f"（{k} 条）标为「{TIER_TOP}」。")
    if compact:
        return (head + f" 前瞻验证：前 10% 的高活性命中率约为随机的 "
                f"{v['EF10_MIC4']:.2f} 倍（MIC ≤ {4} µM）/ "
                f"{v['EF10_MIC1']:.2f} 倍（MIC ≤ 1 µM）。"
                "用途是决定先合成哪几条，不是判定绝对 MIC。")

    lines = [
        head,
        "",
        "这条排序有多可信（下文数字均来自论文前瞻时间切分，"
        f"训练 ≤2016 / 测试 >2016，n = {v['n_test']}）：",
        f"  · 前 10% 的高活性命中率 = 随机的 {v['EF10_MIC4']:.2f} 倍"
        f"（MIC ≤ 4 µM，基础率 {v['base_MIC4']*100:.1f}%）、"
        f"{v['EF10_MIC1']:.2f} 倍（MIC ≤ 1 µM，基础率 {v['base_MIC1']*100:.1f}%）。",
        f"  · 作为对照：仅按结构相似度排序为 {v['baseline_similarity']:.2f}"
        f"（≈随机），仅按菌株中位值为 {v['baseline_strain_median']:.2f}。",
        f"  · 上限：1/基础率 = {v['ceiling_MIC4']:.2f} / {v['ceiling_MIC1']:.2f}，"
        f"模型已达 80% / 54%。",
        f"  · 但前 10% 只覆盖 {v['recall_MIC4']*100:.1f}% / "
        f"{v['recall_MIC1']*100:.1f}% 的真实高活性分子——"
        "它是「先做哪几条」的指南，不是「只做这几条」的许可。",
        "",
        "使用边界：",
        "  1) 排序按预测 MIC 升序；未提供菌株时无定量值，已退化为按定性概率排序。",
        "  2) 富集依赖菌株先验：若该菌株在训练集中样本很少或留出 R² < 0，"
        "排序增益会明显下降（界面菌株下拉已标注支撑度）。",
        "  3) 绝对 MIC 的决策仍应看 90% 区间与信赖等级，而非排序名次。",
    ]
    return "\n".join(lines)
