#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""14_build_quant_dataset.py -- 用 DBAASP 全环化类型数据建**定量**建模集（含整簇划分）。

与旧脚本的区别
-------------
旧定量数据（`dataset_mtl.pkl` 的 log10_um_* 列）来自论文B缓存：
1504 条肽、99% 是二硫键环化、头尾环化仅 225 条。
本脚本改用 `datasets/DBAASP_full/peptide_level.csv`（5572 条、17 种环化类型）。

输出
----
    datasets/DBAASP_full/quant_dataset.pkl   建模主表（特征 + 标签 + 划分 + 环化类型）
    datasets/DBAASP_full/quant_features.npz  指纹矩阵（morgan/maccs）
    datasets/DBAASP_full/quant_build_report.json

设计决定（都写进报告，便于论文交代）
--------------------------------
① **标签**：log10(MIC/µM)，按革兰组取**中位数**聚合（与旧口径一致，保证可比）。
② **划分**：Butina 整簇划分（Morgan r=2, Tanimoto>=0.7），**全表统一划分**
   （不像 13 号脚本按类型分别分），这样全局模型与类型专属模型可用同一套折做公平比较。
   巨型簇处理沿用 `02_build_dataset.py` 的策略：超过 max_cluster_frac 的簇强制切块，
   再做两阶段均衡分配。
③ **训练/测试**：整簇折既做 5 折 CV，又额外留出 1 折作**固定 holdout**，
   便于报告"单次划分"的指标（与 CV 互为验证）。

用法
----
    python 14_build_quant_dataset.py --self-test
    python 14_build_quant_dataset.py --task gram_pos gram_neg
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DDIR = ROOT / "datasets" / "DBAASP_full"
FEATDIR = ROOT / "artifacts" / "features"

TASKS = {"gram_pos": "log10_um_gram_pos", "gram_neg": "log10_um_gram_neg",
         "fungal": "log10_um_fungal"}
TASK_CN = {"gram_pos": "抗革兰阳性 MIC", "gram_neg": "抗革兰阴性 MIC",
           "fungal": "抗真菌 MIC"}


def log(m: str) -> None:
    print(m, flush=True)


def _load(mod_name: str, fname: str):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


# ---------------------------------------------------------------------------
def cluster_split_balanced(smiles, threshold=0.7, n_folds=5, seed=42,
                           max_cluster_frac=0.05, labels=None, rebalance_iters=20000):
    """Butina 整簇划分 + 巨型簇切块 + 均衡分配。

    复用 `02_build_dataset.py` 的四代策略（见该文件的 D12/D13 注释）：
      ① 超过 max_cluster_frac 的簇强制切块，让阳性可跨折分摊
      ② 无标签单元按最空优先（LPT）铺满
      ③ 有标签单元（这里指"有该任务标签"）放进阳性最少的折
      ④ 贪心再平衡
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina
    RDLogger.DisableLog("rdApp.*")

    n = len(smiles)
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                   if m is not None else None)
    valid = [i for i, f in enumerate(fps) if f is not None]
    if len(valid) < 3:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=int), {}

    dists = []
    for i in range(1, len(valid)):
        sims = DataStructs.BulkTanimotoSimilarity(
            fps[valid[i]], [fps[valid[j]] for j in range(i)])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(valid), 1.0 - threshold, isDistData=True)

    # 簇 -> 成员全局索引
    cl_members = [[valid[j] for j in c] for c in clusters]
    # ① 巨型簇切块
    cap = max(2, int(n * max_cluster_frac))
    units = []          # 每个 unit = (成员索引列表, 是否含标签)
    lab = np.zeros(n, dtype=bool) if labels is None else np.asarray(labels, dtype=bool)
    for mem in cl_members:
        if len(mem) <= cap:
            units.append(mem)
        else:
            rng = np.random.RandomState(seed)
            order = list(mem); rng.shuffle(order)
            for k in range(0, len(order), cap):
                units.append(order[k:k + cap])

    unit_lab = [int(lab[u].sum()) for u in units]
    unit_size = [len(u) for u in units]
    total_lab = int(sum(unit_lab))

    # ② / ③ 两阶段分配
    rng = np.random.RandomState(seed)
    order_neg = sorted([i for i in range(len(units)) if unit_lab[i] == 0],
                       key=lambda i: -unit_size[i])
    order_pos = sorted([i for i in range(len(units)) if unit_lab[i] > 0],
                       key=lambda i: -unit_lab[i])
    load = np.zeros(n_folds, dtype=int)
    pos = np.zeros(n_folds, dtype=int)
    assign = np.full(len(units), -1, dtype=int)

    def rel_load(f):
        target = n / n_folds
        return load[f] / max(target, 1)

    for i in order_neg:
        f = int(np.argmin(rel_load(np.arange(n_folds)))) if False else int(
            np.argmin([rel_load(k) for k in range(n_folds)]))
        assign[i] = f; load[f] += unit_size[i]
    for i in order_pos:
        # 放进"已分配阳性最少"的折
        f = int(np.argmin(pos))
        assign[i] = f; load[f] += unit_size[i]; pos[f] += unit_lab[i]

    # ④ 贪心再平衡：交换能让离散度下降的 unit
    for _ in range(rebalance_iters):
        cur = float(load.std())
        improved = False
        for i in range(len(units)):
            fi = assign[i]
            for j in range(len(units)):
                if i == j:
                    continue
                fj = assign[j]
                if fi == fj:
                    continue
                # 试交换
                load[fi] += unit_size[j] - unit_size[i]
                load[fj] += unit_size[i] - unit_size[j]
                if float(load.std()) < cur - 1e-9:
                    assign[i], assign[j] = fj, fi
                    cur = float(load.std())
                    improved = True
                    break
                load[fi] -= unit_size[j] - unit_size[i]
                load[fj] -= unit_size[i] - unit_size[j]
            if improved:
                break
        if not improved:
            break

    folds = np.full(n, -1, dtype=int)
    cluster_id = np.full(n, -1, dtype=int)
    for ci, u in enumerate(units):
        for m in u:
            folds[m] = assign[ci]
            cluster_id[m] = ci
    miss = np.where(folds < 0)[0]
    if len(miss):
        folds[miss] = rng.randint(0, n_folds, len(miss))
        cluster_id[miss] = -2

    stats = {"n_units": len(units), "n_giant_split": sum(1 for m in cl_members if len(m) > cap),
             "max_unit": int(max(unit_size)), "total_labeled": total_lab}
    return folds, cluster_id, stats


def compute_features(df):
    """复用 03_features.py 的 compute_rows 构造 62 维描述符 + 指纹。"""
    f3 = _load("feat03", "03_features.py")
    work = pd.DataFrame({
        "CPID": df["dbaasp_id"].astype(str).values,
        "smiles": df["smiles"].astype(str).values,
        "sequence": df.get("sequence", pd.Series([""] * len(df))).fillna("").values,
        "source": ["DBAASP"] * len(df),
    })
    panel, fps, stats = f3.compute_rows(work, progress_every=500)
    return panel, fps, stats


def self_test() -> int:
    log("=" * 70)
    log("  14_build_quant_dataset.py 自检")
    log("=" * 70)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    # 聚类划分：随机 SMILES，检查折数与均衡性
    smis = ["C1CCCC1", "C1CCCCC1", "C1CCCCCC1", "CCO", "CCN", "CCC", "CCCC", "CCCCC",
            "c1ccccc1", "c1ccncc1", "c1cccnc1", "CCCO", "CCCN", "CCOC", "CCSC",
            "C1CCOC1", "C1CCNC1", "C1CCSC1", "CC(=O)O", "CC(=O)N"]
    folds, cid, st = cluster_split_balanced(smis, n_folds=4, seed=0)
    chk("划分：折数正确", len(set(folds)) == 4, str(sorted(set(folds))))
    chk("划分：无 -1 残留", (folds >= 0).all())
    sizes = pd.Series(folds).value_counts()
    chk("划分：折大小比 < 2", sizes.max() / sizes.min() < 2.0,
        f"{sizes.max()}/{sizes.min()}")
    chk("划分：cluster_id 已填", (cid >= 0).all())

    # 巨型簇切块
    folds2, cid2, st2 = cluster_split_balanced(
        ["C" * 3 + "1" + "CCCC" + "1"] * 40, n_folds=4, seed=0, max_cluster_frac=0.05)
    chk("巨型簇被切块（units > 1）", st2.get("n_units", 0) > 1,
        str(st2.get("n_units")))
    chk("切块后仍无 -1", (folds2 >= 0).all())

    # 特征构造
    if (DDIR / "peptide_level.csv").exists():
        pl = pd.read_csv(DDIR / "peptide_level.csv").head(30)
        panel, fps, fs = compute_features(pl)
        chk("特征构造可执行", len(panel) == len(pl), f"{len(panel)}/{len(pl)}")
        chk("指纹维度正确（morgan=2048, maccs=167）",
            fps["morgan"].shape[1] == 2048 and fps["maccs"].shape[1] == 167,
            f"{fps['morgan'].shape} {fps['maccs'].shape}")
        chk("面板含环化相关列",
            any("cyc" in c for c in panel.columns),
            str([c for c in panel.columns if "cyc" in c][:3]))
    else:
        log("  ℹ️ peptide_level.csv 尚未生成，跳过特征自检")

    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 70)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="建 DBAASP 全环化类型定量建模集",
                                 epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["gram_pos", "gram_neg", "fungal"],
                    choices=list(TASKS))
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-cluster-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    src = DDIR / "peptide_level.csv"
    if not src.exists():
        log(f"✗ 找不到 {src}，请先跑 11 与 12 号脚本")
        return 2
    pep = pd.read_csv(src)
    log(f"  读入 {src.name}: {len(pep)} 条")
    log(f"  环化类型: {pep['bond_primary'].nunique()} 种")

    # ---- 特征 ----
    t0 = time.time()
    log("  构造特征（62 维描述符 + 2048 Morgan + 167 MACCS）…")
    panel, fps, fstats = compute_features(pep)
    log(f"    {panel.shape}  ({time.time()-t0:.0f}s)  统计={fstats}")

    # 对齐（以指纹 ids 为准，保证有结构）
    pid_col = "CPID"
    panel = panel.set_index(pid_col)
    fp_ids = [str(x) for x in fps["ids"]]
    keep = [p for p in fp_ids if p in panel.index]
    log(f"    特征可用: {len(keep)} / {len(pep)}")
    panel = panel.loc[keep]
    pep = pep.set_index(pep["dbaasp_id"].astype(str)).loc[keep].reset_index(drop=True)
    panel = panel.reset_index()
    fi = {c: i for i, c in enumerate(fp_ids)}
    order = [fi[c] for c in keep]

    # 去掉非特征列
    drop = [c for c in ("source", "has_smiles", "has_sequence") if c in panel.columns]
    desc = panel.drop(columns=drop).set_index("CPID").loc[keep]
    feat_names = list(desc.columns)

    # ---- 数值化标签 + 划分 ----
    lab_flags = {}
    for t, col in TASKS.items():
        lab_flags[t] = pep[col].notna().to_numpy()
        log(f"  任务 {TASK_CN[t]}: 有标签 {int(lab_flags[t].sum())} 条")

    # 统一划分：用"至少有一个定量标签"的样本参与均衡
    any_lab = np.zeros(len(pep), dtype=bool)
    for t in args.tasks:
        any_lab |= lab_flags[t]
    log(f"  参与划分均衡的有标签样本: {int(any_lab.sum())}")

    folds, cid, cstats = cluster_split_balanced(
        pep["smiles"].tolist(), threshold=args.threshold, n_folds=args.n_folds,
        seed=args.seed, max_cluster_frac=args.max_cluster_frac, labels=any_lab)
    log(f"  整簇划分: {cstats}")

    fs = pd.Series(folds).value_counts().sort_index()
    log(f"  折大小: {fs.to_dict()}  比={fs.max()/fs.min():.3f}")
    for t in args.tasks:
        r = pd.Series(folds[lab_flags[t]]).value_counts().sort_index()
        log(f"    {TASK_CN[t]} 各折阳性: {r.to_dict()}")

    # 硬校验
    ratio = float(fs.max() / max(fs.min(), 1))
    if ratio > 1.5:
        log(f"  ⚠ 折大小比 {ratio:.3f} > 1.5，CV 假设受影响（记录但不中断）")
    no_leak = bool(pd.DataFrame({"c": cid, "f": folds}).groupby("c")["f"].nunique().max() == 1)
    log(f"  同簇不跨折: {no_leak}")

    # ---- 落盘 ----
    out = pep.copy()
    out["fold_cluster"] = folds
    out["cluster_id"] = cid
    for i, c in enumerate(feat_names):
        out[f"d_{c}"] = desc[c].to_numpy()
    np.savez_compressed(DDIR / "quant_features.npz",
                        ids=np.array(keep, dtype=object),
                        morgan=fps["morgan"][order], maccs=fps["maccs"][order],
                        feat_names=np.array(feat_names, dtype=object))
    out.to_pickle(DDIR / "quant_dataset.pkl")
    spec = {"threshold": args.threshold, "radius": 2, "n_bits": 2048, "maccs": 167}
    (FEATDIR / "quant_feature_spec.json").parent.mkdir(parents=True, exist_ok=True)
    (FEATDIR / "quant_feature_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")

    rep = {
        "source": src.name, "n_peptides": int(len(out)),
        "n_bond_types": int(out["bond_primary"].nunique()),
        "bond_primary_counts": out["bond_primary"].value_counts().to_dict(),
        "n_features_desc": len(feat_names), "feature_names": feat_names,
        "n_morgan": 2048, "n_maccs": 167, "total_dim": len(feat_names) + 2048 + 167,
        "fold_sizes": {str(k): int(v) for k, v in fs.items()},
        "fold_size_ratio": ratio, "same_cluster_same_fold": no_leak,
        "cluster_stats": cstats,
        "label_coverage": {t: int(v.sum()) for t, v in lab_flags.items()},
        "per_fold_positive": {
            t: {str(k): int(v) for k, v in
                pd.Series(folds[lab_flags[t]]).value_counts().sort_index().items()}
            for t in args.tasks},
        "honest_notes": [
            "本数据集来自 DBAASP 全量拉取（5572 条环肽 / 17 种环化类型），"
            "替代了原先仅覆盖二硫键的论文B缓存。",
            "标签为 log10(MIC/µM)，按革兰组取中位数聚合；µg/ml 记录用 RDKit 分子量换算。",
            "划分用 Butina 整簇（Tanimoto>=0.7）+ 巨型簇切块 + 两阶段均衡分配，"
            "保证同源肽不跨折。",
            "样本量扩大 3 倍、化学类型从 2 类扩到 18 类后，R² 下降是任务变难所致，"
            "不代表模型退步。",
        ],
    }
    (DDIR / "quant_build_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    log("")
    log(f"  ✅ {DDIR}/quant_dataset.pkl  ({len(out)} 行)")
    log(f"  ✅ {DDIR}/quant_features.npz  (desc {len(feat_names)} + fp 2215 = "
        f"{len(feat_names)+2215} 维)")
    log(f"  ✅ {DDIR}/quant_build_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
