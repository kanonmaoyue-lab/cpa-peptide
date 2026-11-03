#!/usr/bin/env python -X utf8
# -*- coding: utf-8 -*-
"""69_pantarget_qualitative.py -- 定性的"统一模型 + 目标条件化"。

动机
----
定量侧的成功（`跨菌株统一模型_结果.md`）：
    **1 个统一模型 + 菌株上下文（R² 0.573）> 6 个逐菌株模型（0.553）**
    且统一模型是**唯一可用于新菌株**的形式。

⭐ 定性侧可以做完全平行的改造：
    现状 = **5 个二分类任务各自独立建模**（抗细菌 / 阳性 / 阴性 / 真菌 / 总抗菌）
    改造 = **1 个统一模型 + 目标条件化**（target conditioning）

**为什么可能更好**：
    · 5 个任务共享同一批肽 → 目标间信息互补
    · **抗真菌只有 415 条正例（最弱，MCC 0.5547）** → 统一模型可借其他目标的信息
    · 单药的"广谱性"天然是相关的（抗阳性与抗阴性有共现结构）

设计
----
    把 (肽, 目标) 对当作样本：
        肽特征 2277 维 + **目标 one-hot 5 维** → 1 个二分类器
    样本数 = 1769 × 5 = 8845（实际按有标签的对计）

    四个对比
        Q0  5 个独立模型（现状基线；每任务 LGBM）
        Q1  ⭐ 统一 + 目标 one-hot（**本文提出**）
        Q2  统一 + 目标 one-hot + 任务间共现特征（该肽在其它 4 个目标上的预测）
        Q3  5 个独立模型但**多任务标签共享**（把其它目标标签作为特征 —— 探索上界）

    ⚠️ Q3 用标签作特征会泄漏（同一肽的其它标签与目标标签相关），
       故仅作"上界参考"，明确标注为**不诚实口径**。

评估
----
    · 整簇 5 折（与原口径一致）
    · **逐目标 AUROC / MCC**，与现状基线（`05_model_baseline.py`）对照
    · 跨来源外部验证（V1，若有）

用法
----
    python 69_pantarget_qualitative.py --self-test
    python 69_pantarget_qualitative.py --run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parent
DDIR = ROOT / "datasets"
OUTD = ROOT / "artifacts" / "pantarget"
try:
    OUTD.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

#: 5 个定性目标（列名与 05 号一致）
TASKS = [("总抗菌(H2)", "anti_bacterial_any"),
         ("抗细菌", "anti_bacterial"),
         ("抗革兰阳性", "anti_gram_pos"),
         ("抗革兰阴性", "anti_gram_neg"),
         ("抗真菌", "anti_fungal")]


def log(m: str) -> None:
    print(m, flush=True)


def _load(mod_name: str, fname: str):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def binarize(s):
    if s.dtype == bool:
        return s.astype(int)
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() == 0:
        m = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        v = s.astype(str).str.strip().str.lower().map(m)
    return v


def load_data(sequence_mode="resolved_v2"):
    """载入 2277 维定性特征 + 5 个标签。"""
    mtl = pd.read_pickle(DDIR / "dataset_mtl.pkl")
    ann = mtl[mtl["label_source"].astype(str) == "annotated"].copy()
    ann = ann.dropna(subset=["smiles"])
    ann = ann[ann["smiles"].astype(str).str.len() > 2]
    ann["CPID"] = ann["CPID"].astype(str)
    ann = ann.drop_duplicates("CPID", keep="first")

    if sequence_mode not in ("legacy", "resolved_v2"):
        raise ValueError("sequence_mode must be legacy or resolved_v2")
    from cpamp_sequence import resolve_qualitative, replace_sequence_block
    resolved = [resolve_qualitative(row)[0] for _, row in ann.iterrows()]
    sequence_map = dict(zip(ann["CPID"], resolved))
    f3 = _load("f3", "03_features.py")
    work = pd.DataFrame({"CPID": ann["CPID"].values,
                         "smiles": ann["smiles"].astype(str).values,
                         "sequence": [""] * len(ann),
                         "source": ["qual"] * len(ann)})
    panel, fps, _ = f3.compute_rows(work, progress_every=0)
    pan = panel.set_index("CPID")
    qd = pd.read_pickle(DDIR / "DBAASP_full" / "quant_dataset.pkl")
    dnames = [c[2:] for c in qd.columns if c.startswith("d_")]
    fid = {str(c): i for i, c in enumerate(fps["ids"])}
    keep, X = [], []
    for c in work["CPID"]:
        if c not in pan.index or c not in fid:
            continue
        v = (pd.to_numeric(pan.loc[c].reindex(dnames), errors="coerce")
             .astype(float).to_numpy())
        i = fid[c]
        # ⚠️ 坑：描述符有 31 列缺失（62 维里有 NaN），必须显式填补。
        #    05 号用的是 `np.nan_to_num(arr, nan=0.0)`；本脚本第一版**漏了**
        #    → 全部 1405 行含 NaN → 自检"同一肽特征一致"报错（NaN != NaN 恒真）。
        #    注意：LightGBM 能静默吃 NaN，所以这个 bug 不会自己暴露，
        #    只会让结果与 05 号不可比。统一按 05 号口径填 0。
        arr = np.concatenate([v, fps["morgan"][i].astype(float),
                              fps["maccs"][i].astype(float)])
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        X.append(arr)
        keep.append(c)
    X = np.vstack(X).astype(np.float32)
    if sequence_mode == "resolved_v2":
        X = replace_sequence_block(X, [sequence_map[c] for c in keep])
    sub = ann.set_index("CPID").loc[keep]
    Y = {t: binarize(sub[c]).fillna(0).astype(int).to_numpy()
         for t, c in TASKS if c in sub.columns}
    return keep, X, Y, sub


def long_format(X, Y):
    """把宽表 (n_pep, n_task) 展开成长表 (n_pep × n_task, d + n_task)。"""
    tasks = list(Y)
    rows, ys, tids, pids = [], [], [], []
    for j, t in enumerate(tasks):
        n = X.shape[0]
        oh = np.zeros((n, len(tasks)), dtype=np.float32)
        oh[:, j] = 1.0
        rows.append(np.hstack([X, oh]))
        ys.append(Y[t])
        tids.append(np.full(n, j))
        pids.append(np.arange(n))
    return (np.vstack(rows), np.concatenate(ys), np.concatenate(tids),
            np.concatenate(pids), tasks)


def self_test() -> int:
    log("=" * 78)
    log("  69_pantarget_qualitative.py 自检")
    log("=" * 78)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    chk("定性数据存在", (DDIR / "dataset_mtl.pkl").exists())
    chk("5 个目标定义", len(TASKS) == 5, str(len(TASKS)))
    try:
        keep, X, Y, sub = load_data()
        chk("数据可加载", X.shape[0] > 1000 and X.shape[1] == 2277,
            f"{X.shape}")
        # v2: missing residue descriptors stay NaN for native LightGBM handling.
        # Structural descriptors/fingerprints remain finite; no infinities anywhere.
        chk("⭐ 结构特征有限且无 Inf",
            bool(np.isfinite(X[:, 31:]).all() and not np.isinf(X).any()),
            f"已标记缺失序列值={int(np.isnan(X[:, :31]).sum())}")
        chk("⭐ 31列序列特征均恢复变化",
            bool(np.all(np.nanstd(X[:, :31], axis=0) > 0)))
        chk("5 个标签齐备", len(Y) == 5, str(list(Y)))
        for t, _ in TASKS:
            if t in Y:
                p = int(Y[t].sum())
                chk(f"ⓘ {t} 正例 {p}", p > 100, str(p))
        # ⭐ 长表格式
        Xl, yl, tl, pl, tasks = long_format(X, Y)
        chk("长表行数 = 肽数 × 5", Xl.shape[0] == X.shape[0] * 5,
            f"{Xl.shape[0]} vs {X.shape[0]*5}")
        chk("长表维度 = 2277 + 5", Xl.shape[1] == 2277 + 5, str(Xl.shape[1]))
        # ⭐ one-hot 正确性
        oh = Xl[:, 2277:]
        chk("⭐ 目标 one-hot 每行恰 1 个为 1",
            bool(np.all(oh.sum(axis=1) == 1)), str(np.unique(oh.sum(axis=1))))
        chk("⭐ one-hot 与目标编号一致",
            bool(np.all(oh[np.arange(len(tl)), tl] == 1)))
        # ⭐ 同一肽在 5 个目标下的肽特征应完全相同
        same = True
        for j in range(1, 5):
            a = Xl[pl == 0][0, :2277]
            b = Xl[pl == 0][j, :2277]
            same &= bool(np.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True))
        chk("⭐ 同一肽的肽特征在各目标下一致", same)
        chk("yl 长度正确", len(yl) == Xl.shape[0])
    except Exception as e:  # noqa: BLE001
        import traceback
        chk("数据可加载", False, f"{type(e).__name__}: {str(e)[:70]}")
        log("   " + traceback.format_exc()[-400:].replace("\n", "\n   "))
    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 78)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def run(seeds=(0, 1, 2), sequence_mode="resolved_v2"):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    t0 = time.time()
    keep, X, Y, sub = load_data(sequence_mode=sequence_mode)
    tasks = [t for t, _ in TASKS if t in Y]
    log("")
    log("=" * 110)
    log(f"  定性统一模型：{X.shape[0]} 条肽 × {len(tasks)} 个目标")
    log("=" * 110)

    # 折：从 dataset_mtl 的 fold_cluster（若有），否则自建
    if "fold_cluster" in sub.columns:
        fold = sub["fold_cluster"].to_numpy()
    else:
        m14 = _load("m14", "14_build_quant_dataset.py")
        fl, _, _ = m14.cluster_split_balanced(
            sub["smiles"].astype(str).tolist(), threshold=0.7, n_folds=5,
            seed=42, labels=np.ones(len(sub), dtype=bool))
        fold = fl
    log(f"  整簇 5 折：{pd.Series(fold).value_counts().to_dict()}")

    Xl, yl, tl, pl, tlist = long_format(X, {t: Y[t] for t in tasks})
    # 长表的折 = 肽的折
    fl = fold[pl]

    rows = []

    # ---------- Q0 逐目标独立（现状基线，口径一致） ----------
    log("")
    log("  【Q0】5 个独立模型（现状基线，本脚本重算以保证口径一致）")
    q0 = {}
    for j, t in enumerate(tasks):
        m = tl == j
        oof = np.full(int(m.sum()), np.nan)
        fo = fold
        for f in sorted(set(fo)):
            te = fo == f
            tr = ~te
            if tr.sum() < 40 or te.sum() < 5 or len(set(Y[t][tr])) < 2:
                continue
            ps = []
            for sd in seeds:
                try:
                    mdl = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                             num_leaves=31, random_state=sd + int(f),
                                             verbose=-1)
                    mdl.fit(X[tr], Y[t][tr])
                    ps.append(mdl.predict_proba(X[te])[:, 1])
                except Exception:  # noqa: BLE001
                    pass
            if ps:
                oof[te] = np.mean(ps, axis=0)
        ok = np.isfinite(oof)
        if ok.sum() < 20 or len(set(Y[t][ok])) < 2:
            continue
        pred = (oof[ok] >= 0.5).astype(int)
        q0[t] = {"AUROC": float(roc_auc_score(Y[t][ok], oof[ok])),
                 "MCC": float(matthews_corrcoef(Y[t][ok], pred)),
                 "n_pos": int(Y[t][ok].sum()), "n": int(ok.sum())}
        rows.append({"model": "Q0 逐目标独立", "task": t, **q0[t]})

    # ---------- Q1 统一 + 目标 one-hot ----------
    log("")
    log("  【Q1】⭐ 统一模型 + 目标 one-hot（1 个模型学全部 8845 个对）")
    oof_l = np.full(len(yl), np.nan)
    for f in sorted(set(fl)):
        te = fl == f
        tr = ~te
        if tr.sum() < 200 or te.sum() < 20:
            continue
        ps = []
        for sd in seeds:
            try:
                mdl = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                         num_leaves=31, random_state=sd + int(f),
                                         verbose=-1)
                mdl.fit(Xl[tr], yl[tr])
                ps.append(mdl.predict_proba(Xl[te])[:, 1])
            except Exception:  # noqa: BLE001
                pass
        if ps:
            oof_l[te] = np.mean(ps, axis=0)
    q1 = {}
    for j, t in enumerate(tasks):
        m = tl == j
        v = oof_l[m]
        ok = np.isfinite(v)
        if ok.sum() < 20 or len(set(Y[t][ok])) < 2:
            continue
        pred = (v[ok] >= 0.5).astype(int)
        q1[t] = {"AUROC": float(roc_auc_score(Y[t][ok], v[ok])),
                 "MCC": float(matthews_corrcoef(Y[t][ok], pred)),
                 "n_pos": int(Y[t][ok].sum()), "n": int(ok.sum())}
        rows.append({"model": "Q1 统一+目标one-hot", "task": t, **q1[t]})

    # ---------- 报告 ----------
    log("")
    log(f"    {'目标':<14}{'Q0 AUROC':>11}{'Q1 AUROC':>11}{'Δ':>9}"
        f"{'Q0 MCC':>10}{'Q1 MCC':>10}{'Δ':>9}{'正例':>7}")
    log("    " + "-" * 82)
    for t in tasks:
        a, b = q0.get(t), q1.get(t)
        if not a or not b:
            continue
        da = b["AUROC"] - a["AUROC"]
        dm = b["MCC"] - a["MCC"]
        log(f"    {t:<14}{a['AUROC']:>11.4f}{b['AUROC']:>11.4f}{da:>+9.4f}"
            f"{a['MCC']:>10.4f}{b['MCC']:>10.4f}{dm:>+9.4f}{b['n_pos']:>7}")

    D = pd.DataFrame(rows)
    output_dir = OUTD if sequence_mode == "legacy" else ROOT / "artifacts" / "sequence_repair_v2"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    fp = output_dir / ("pantarget_metrics.csv" if sequence_mode == "legacy"
                       else "independent_vs_unified_corrected.csv")
    if fp.exists():
        old = pd.read_csv(fp)
        D = pd.concat([old, D], ignore_index=True).drop_duplicates(
            subset=["model", "task"], keep="last")
    D.to_csv(fp, index=False, encoding="utf-8-sig")

    log("")
    log("=" * 110)
    log("  ⭐ 汇总")
    log("=" * 110)
    g = D.groupby("model").agg(AUROC=("AUROC", "mean"), MCC=("MCC", "mean")).reset_index()
    log(f"    {'模型':<24}{'平均 AUROC':>12}{'平均 MCC':>11}")
    log("    " + "-" * 48)
    for _, r in g.iterrows():
        log(f"    {r['model']:<24}{r['AUROC']:>12.4f}{r['MCC']:>11.4f}")
    b0 = g[g["model"].str.startswith("Q0")]
    if len(b0):
        for _, r in g.iterrows():
            if r["model"].startswith("Q0"):
                continue
            log(f"    → {r['model']:<22}"
                f"ΔAUROC={r['AUROC']-b0['AUROC'].iloc[0]:+.4f}  "
                f"ΔMCC={r['MCC']-b0['MCC'].iloc[0]:+.4f}")
    log("")
    log(f"  用时 {time.time()-t0:.0f}s")
    log(f"  ✅ 已写出 {fp}")
    return D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="定性统一模型 + 目标条件化")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sequence-mode", choices=["legacy", "resolved_v2"],
                    default="resolved_v2")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.run:
        run(seeds=tuple(args.seeds), sequence_mode=args.sequence_mode)
        return 0
    log("请指定 --run 或 --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
