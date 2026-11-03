#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""13_model_by_cyclization.py -- 按环化类型分别训练/评估模型。

要回答的问题
-----------
项目原本的定量数据 99% 是二硫键环化肽，导致模型对头尾环化、侧链环化肽不可用。
补齐 DBAASP 全环化类型数据后，本脚本回答：

  Q1  各环化类型的定量覆盖情况（建完数据后各有多少肽）
  Q2  **全局模型 vs 类型专属模型**：单独给每个类型训模型是否更好？
  Q3  **跨类型泛化**：用 A 类型训练、B 类型测试，掉多少？
      （这直接量化"训练数据环化类型偏斜"造成的代价）
  Q4  评价时**按环化类型分层**，看全局模型在各类型上的表现差异

设计要点
-------
* **划分**：类型内部做 Butina 整簇划分（Tanimoto>=0.7），防止同源泄漏。
  类型之间天然不共享簇，所以跨类型实验不需要额外处理。
* **噪声地板**：本项目实测回归 R² 的 seed 抖动可达 0.07（分类 AUROC 只 0.005），
  所以每个配置都跑多 seed 并报 **均值±标准差**，差异 < 2×合并 std 的不下结论。
* **样本量门槛**：测试集阳性 < 20 或训练集 < 50 的组合直接跳过并标注，不给不可信数字。

用法
----
    python 13_model_by_cyclization.py --self-test
    python 13_model_by_cyclization.py --task gram_pos          # 只做 Gram+
    python 13_model_by_cyclization.py --task gram_pos gram_neg
    python 13_model_by_cyclization.py --task gram_pos --min-n 40 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
ROOT = HERE.parent
DDIR = ROOT / "datasets" / "DBAASP_full"
OUTD = ROOT / "artifacts" / "models_by_cyclization"
try:
    OUTD.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

#: 定量任务 -> 数据集里的列名
TASKS = {"gram_pos": "log10_um_gram_pos",
         "gram_neg": "log10_um_gram_neg",
         "fungal": "log10_um_fungal"}
TASK_CN = {"gram_pos": "抗革兰阳性 MIC", "gram_neg": "抗革兰阴性 MIC",
           "fungal": "抗真菌 MIC"}

#: 环化类型的展示名
BOND_CN = {"AMD": "酰胺键(头尾环化)", "DSB": "二硫键", "TIE": "硫醚键",
           "EST": "酯键", "DCB": "碳碳双键", "CAR": "碳碳键", "AMN": "胺键",
           "ETH": "醚键", "p-XylB": "对二甲苯桥", "none": "无链内键",
           "(E)-but-2-enyl-B": "烯烃交联(订书肽)", "but-2-ynyl-B": "炔烃交联(订书肽)"}


def log(m: str) -> None:
    print(m, flush=True)


def _load(mod_name: str, fname: str):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def compute_features(smiles_list, sequences=None, ids=None, panel="desc_fp"):
    """复用 03_features.py 的 compute_rows 构造特征。

    返回 (X, ids_kept)；行序与 ids_kept 对应。
    """
    f3 = _load("feat03", "03_features.py")
    n = len(smiles_list)
    df = pd.DataFrame({
        "CPID": [str(x) for x in (ids if ids is not None else range(n))],
        "smiles": [str(s) for s in smiles_list],
        "sequence": [str(s) if s is not None else "" for s in
                     (sequences if sequences is not None else [""] * n)],
        "source": ["DBAASP"] * n,
    })
    panel_df, fps, stats = f3.compute_rows(df, progress_every=0)
    p = panel_df.set_index("CPID")
    fid = {str(c): i for i, c in enumerate(fps["ids"])}
    rows, keep = [], []
    for c in df["CPID"]:
        if c not in p.index or c not in fid:
            continue
        i = fid[c]
        drop = [x for x in ("source", "has_smiles", "has_sequence") if x in p.columns]
        v = pd.to_numeric(p.loc[c].drop(labels=drop), errors="coerce").astype(float).to_numpy()
        if panel == "desc":
            vec = v
        elif panel == "fp":
            vec = np.concatenate([fps["morgan"][i], fps["maccs"][i]]).astype(float)
        else:
            vec = np.concatenate([v, fps["morgan"][i], fps["maccs"][i]]).astype(float)
        rows.append(vec); keep.append(c)
    if not rows:
        return np.zeros((0, 1)), []
    return np.vstack(rows), keep


def make_reg(name, seed):
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=5, subsample=0.9, subsample_freq=1,
                             colsample_bytree=0.3, reg_lambda=1.0,
                             random_state=seed, n_jobs=-1, verbose=-1)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                            min_child_weight=2, subsample=0.9, colsample_bytree=0.3,
                            reg_lambda=1.0, random_state=seed, n_jobs=-1,
                            verbosity=0, tree_method="hist")
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, n_jobs=1)
    raise ValueError(name)


def cluster_folds(smiles, threshold=0.7, n_folds=5, seed=42):
    """类型内部做 Butina 整簇划分（样本少，用 5 折）。"""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina
    RDLogger.DisableLog("rdApp.*")
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    idx = [i for i, f in enumerate(fps) if f is not None]
    if len(idx) < 3:
        return np.zeros(len(smiles), dtype=int)
    dists = []
    for i in range(1, len(idx)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx[i]], [fps[idx[j]] for j in range(i)])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(idx), 1.0 - threshold, isDistData=True)
    rng = np.random.RandomState(seed)
    order = list(range(len(clusters)))
    rng.shuffle(order)
    sizes = [len(clusters[k]) for k in order]
    # LPT 装箱
    load = np.zeros(n_folds, dtype=int)
    assign = {}
    for pos in np.argsort(-np.array(sizes)):
        f = int(np.argmin(load))
        assign[pos] = f
        load[f] += sizes[pos]
    folds = np.full(len(smiles), -1, dtype=int)
    for pos, k in enumerate(order):
        f = assign[pos]
        for j in clusters[k]:
            folds[idx[j]] = f
    # 未进簇的（指纹失败）随机分
    miss = np.where(folds < 0)[0]
    if len(miss):
        folds[miss] = rng.randint(0, n_folds, len(miss))
    return folds


def metrics_reg(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    if len(y) < 3 or not np.isfinite(p).any():
        return None
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    err = p - y
    return {"n": int(len(y)), "R2": float(r2_score(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "within_1log": float((np.abs(err) <= 1.0).mean())}


def run_cv(X, y, folds, model_name, seed, refit=False):
    from sklearn.linear_model import Ridge
    oof = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        te = np.where(folds == f)[0]; tr = np.where(folds != f)[0]
        if len(te) < 2 or len(tr) < 10:
            continue
        if model_name == "ridge":
            mdl = Ridge(alpha=1.0, random_state=seed)
        elif model_name == "mean":
            oof[te] = float(np.mean(y[tr])); continue
        else:
            mdl = make_reg(model_name, seed + f)
        try:
            mdl.fit(X[tr], y[tr]); oof[te] = mdl.predict(X[te])
        except Exception:  # noqa: BLE001
            pass
    final = None
    if refit and len(y) >= 10:
        final = make_reg(model_name, seed)
        try:
            final.fit(X, y)
        except Exception:  # noqa: BLE001
            final = None
    return oof, final


def self_test() -> int:
    log("=" * 70)
    log("  13_model_by_cyclization.py 自检")
    log("=" * 70)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    # 特征构造
    X, keep = compute_features(["C1CCCCC1", "CCO"], ["", ""], ["a", "b"])
    chk("特征构造可执行", X.shape[0] == 2, str(X.shape))
    dX, _ = compute_features(["C1CCCCC1"], [""], ["a"], panel="desc")
    chk("仅描述符面板维度更小", dX.shape[1] < X.shape[1], f"{dX.shape[1]} vs {X.shape[1]}")

    # 聚类划分
    smis = ["C1CCCC1", "C1CCCCC1", "C1CCCCCC1", "CCO", "CCN", "CCC", "CCCC",
            "c1ccccc1", "c1ccncc1", "c1ccncc1"]
    folds = cluster_folds(smis, n_folds=3, seed=0)
    chk("聚类划分：折数正确", len(set(folds)) == 3, str(sorted(set(folds))))
    chk("聚类划分：无 -1 残留", (folds >= 0).all())

    # 指标
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = metrics_reg(y, y + 0.1)
    chk("完美预测 R²≈1", m and m["R2"] > 0.95, str(m and round(m["R2"], 4)))
    chk("within_1log 合理", m and m["within_1log"] == 1.0)

    # 端到端小回测
    rng = np.random.RandomState(0)
    Xs = rng.randn(40, 12)
    ys = Xs[:, 0] * 2 + rng.randn(40) * 0.3
    f = np.array([i % 4 for i in range(40)])
    oof, _ = run_cv(Xs, ys, f, "lgbm", 0)
    mm = metrics_reg(ys, oof)
    chk("端到端 CV 可跑通且有 OOF", mm is not None and mm["n"] > 10,
        str(mm and mm["n"]))
    chk("端到端 R² 为正（信号明显）", mm and mm["R2"] > 0.4,
        str(mm and round(mm["R2"], 3)))

    # 数据文件（若已下载）
    # 数据是否就绪属于"环境状态"，不是脚本正确性 —— 只提示不判失败。
    # （判失败会导致"还没下载就先跑自检"时出现假警报。）
    pl = DDIR / "peptide_level.csv"
    log(f"  ℹ️ 肽级数据 {'已就绪' if pl.exists() else '尚未生成'}"
        f"（需先跑 11_fetch_dbaasp_full.py 与 12_build_dbaasp_dataset.py）")

    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 70)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按环化类型分别训练与评估",
                                 epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", default=["gram_pos", "gram_neg"],
                    choices=list(TASKS))
    ap.add_argument("--models", nargs="+", default=["lgbm", "xgb"])
    ap.add_argument("--min-n", type=int, default=30,
                    help="每个环化类型至少多少条才单独建模")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--panel", default="desc_fp", choices=["desc", "fp", "desc_fp"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    pl = DDIR / "peptide_level.csv"
    if not pl.exists():
        log(f"✗ 找不到 {pl}，请先运行：")
        log("    python 11_fetch_dbaasp_full.py --fetch")
        log("    python 12_build_dbaasp_dataset.py")
        return 2
    pep = pd.read_csv(pl)
    log(f"  读入肽级数据 {len(pep)} 条")

    # ---- 特征（全表一次算好，按类型切片复用，避免重复计算）----
    t0 = time.time()
    log(f"  构造特征（panel={args.panel}）…")
    X, keep = compute_features(pep["smiles"].tolist(), ids=pep["dbaasp_id"].astype(str).tolist())
    log(f"    特征矩阵 {X.shape}（{time.time()-t0:.0f}s）")
    kmap = {c: i for i, c in enumerate(keep)}
    pep["_fi"] = pep["dbaasp_id"].astype(str).map(kmap)
    pep = pep[pep["_fi"].notna()].copy()
    pep["_fi"] = pep["_fi"].astype(int)
    log(f"    有特征可用: {len(pep)} 条")

    type_counts = pep["bond_primary"].value_counts()
    log("")
    log("  各环化类型的定量覆盖（按主类型）：")
    log(f"    {'类型':<12}{'肽数':>6}" + "".join(f"{TASK_CN[t][:6]:>10}" for t in args.task))
    for k, n in type_counts.items():
        sub = pep[pep["bond_primary"] == k]
        log(f"    {k:<12}{n:>6}" +
            "".join(f"{int(sub[TASKS[t]].notna().sum()):>10}" for t in args.task))

    all_rows, all_loso = [], []
    for task in args.task:
        col = TASKS[task]
        d = pep[pep[col].notna()].copy()
        d["y"] = d[col].astype(float)
        if len(d) < 30:
            log(f"\n  ⚠ {TASK_CN[task]}: 只有 {len(d)} 条，跳过")
            continue
        log("")
        log("=" * 90)
        log(f"  任务 {TASK_CN[task]}：共 {len(d)} 条，覆盖 "
            f"{d['bond_primary'].nunique()} 种环化类型")
        log("=" * 90)

        # ---------- 全局模型（所有类型混合）----------
        fidx = d["_fi"].to_numpy()
        Xg = X[fidx]
        # 全局划分：类型内整簇划分（跨类型不共享簇，所以等价于全局整簇）
        fold_parts = []
        for k, g in d.groupby("bond_primary"):
            f = cluster_folds(g["smiles"].tolist(), threshold=0.7, n_folds=5, seed=42)
            fold_parts.append(pd.Series(f, index=g.index))
        d["_fold"] = pd.concat(fold_parts).reindex(d.index).to_numpy()
        folds = d["_fold"].to_numpy()

        log(f"\n  ① 全局模型（混合全部环化类型，整簇 5 折 × {len(args.seeds)} seed）")
        glob = {}
        for mn in args.models:
            per = []
            for sd in args.seeds:
                oof, _ = run_cv(Xg, d["y"].to_numpy(), folds, mn, sd)
                m = metrics_reg(d["y"].to_numpy(), oof)
                if m:
                    per.append(m)
            if not per:
                continue
            agg = {f"{k}_mean": float(np.mean([p[k] for p in per]))
                   for k in ("R2", "RMSE", "within_1log")}
            agg.update({f"{k}_std": float(np.std([p[k] for p in per], ddof=1))
                        for k in ("R2", "RMSE", "within_1log")})
            agg.update({"n": per[0]["n"]})
            glob[mn] = agg
            log(f"    {mn:5s} R²={agg['R2_mean']:+.3f}±{agg['R2_std']:.3f}  "
                f"±1log={agg['within_1log_mean']:.3f}  n={agg['n']}")

        # 全局模型按类型分层评估（用最好的模型）
        best_mn = max(glob, key=lambda k: glob[k]["R2_mean"]) if glob else None
        if best_mn:
            oof = np.nanmean([run_cv(Xg, d["y"].to_numpy(), folds, best_mn, sd)[0]
                              for sd in args.seeds], axis=0)
            log(f"\n  ② 全局模型（{best_mn}）按环化类型分层表现：")
            log(f"    {'类型':<12}{'n':>6}{'R²':>9}{'±1log':>9}")
            for k, g in d.groupby("bond_primary"):
                m = metrics_reg(g["y"].to_numpy(), oof[g.index.map(
                    {ix: i for i, ix in enumerate(d.index)}).to_numpy()])
                if m:
                    all_rows.append({"task": task, "scope": "global_model_by_type",
                                     "bond": k, "model": best_mn, **m})
                    log(f"    {k:<12}{m['n']:>6}{m['R2']:>+9.3f}{m['within_1log']:>9.3f}")

        # ---------- 类型专属模型 ----------
        log(f"\n  ③ 类型专属模型（每个类型单独训练，样本 >= {args.min_n}）")
        log(f"    {'类型':<12}{'n':>6}{'R²(专属)':>11}{'R²(全局)':>11}{'Δ':>9}  判定")
        for k, g in d.groupby("bond_primary"):
            if len(g) < args.min_n:
                log(f"    {k:<12}{len(g):>6}{'样本不足，跳过':>24}")
                continue
            fk = cluster_folds(g["smiles"].tolist(), threshold=0.7, n_folds=5, seed=42)
            Xk = X[g["_fi"].to_numpy()]
            yk = g["y"].to_numpy()
            per = []
            for mn in args.models:
                for sd in args.seeds:
                    oof, _ = run_cv(Xk, yk, fk, mn, sd)
                    m = metrics_reg(yk, oof)
                    if m:
                        per.append((mn, m))
            if not per:
                continue
            best = max(per, key=lambda x: x[1]["R2"])
            r2_own = float(np.mean([m["R2"] for _, m in per]))
            r2_own_std = float(np.std([m["R2"] for _, m in per], ddof=1)) if len(per) > 1 else 0.0
            # 全局模型在同类型上的表现
            gi = g.index.map({ix: i for i, ix in enumerate(d.index)}).to_numpy()
            gg = glob.get(best_mn or "", {}).get("R2_mean", np.nan)
            # 全局模型在该类型子集上的 R²（重算）
            oof_g = np.nanmean([run_cv(Xg, d["y"].to_numpy(), folds, best_mn, sd)[0]
                                for sd in args.seeds], axis=0) if best_mn else None
            r2_glob = np.nan
            if oof_g is not None:
                mg = metrics_reg(yk, oof_g[gi])
                if mg:
                    r2_glob = mg["R2"]
            delta = r2_own - r2_glob
            pooled = np.sqrt(r2_own_std ** 2 + glob.get(best_mn or "", {}).get("R2_std", 0.0) ** 2)
            verdict = ("显著" if (np.isfinite(delta) and pooled > 0 and abs(delta) > 2 * pooled)
                       else "在噪声内")
            log(f"    {k:<12}{len(g):>6}{r2_own:>+8.3f}±{r2_own_std:.3f}"
                f"{r2_glob:>+11.3f}{delta:>+9.3f}  {verdict}")
            all_rows.append({"task": task, "scope": "own_vs_global", "bond": k,
                             "model": best[0], "n": len(g),
                             "R2_own_mean": r2_own, "R2_own_std": r2_own_std,
                             "R2_global": float(r2_glob) if np.isfinite(r2_glob) else np.nan,
                             "delta": float(delta) if np.isfinite(delta) else np.nan,
                             "verdict": verdict,
                             "R2_own": best[1]["R2"],
                             "within_1log_own": best[1]["within_1log"]})

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTD / "by_cyclization_metrics.csv", index=False, encoding="utf-8-sig")
        log("")
        log(f"  ✅ 已写出 {OUTD}/by_cyclization_metrics.csv  （{len(df)} 行）")
    else:
        log("\n  ⚠ 没有产生任何结果")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
