#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""15_model_quant.py -- 新数据集上的**定量**模型训练与评估（多模型 × 多 seed）。

与 06_model_regression.py 的区别
------------------------------
* 数据源：`datasets/DBAASP_full/quant_dataset.pkl`（4407 条、18 种环化类型）
  而非旧的论文B缓存（1504 条、几乎全是二硫键）。
* 评估：多 seed + 多折，报 **均值±标准差**，并给"是否超噪声"的判定。
* 分层：按环化类型分别报告，量化适用边界。
* 多了 **GBDT / Ridge / 均值基线** 三条对照线。

关键方法学约束（本项目实测教训）
------------------------------
* 回归的 seed 抖动远大于分类（实测 R² std 可达 0.07），
  所以**每个配置必须多 seed**，单 seed 结论不可信。
* 划分用整簇折（同源肽不跨折），随机划分会把指标虚高。
* 必须报**均值基线**对照：R²<0 说明模型还不如直接猜平均值。

用法
----
    python 15_model_quant.py --self-test
    python 15_model_quant.py --task gram_pos gram_neg
    python 15_model_quant.py --task gram_pos --models lgbm xgb rf ridge mean
    python 15_model_quant.py --holdout            # 额外用固定 holdout 折评估
"""
from __future__ import annotations

import argparse
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
OUTD = ROOT / "artifacts" / "quant"
try:
    OUTD.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

TASKS = {"gram_pos": "log10_um_gram_pos", "gram_neg": "log10_um_gram_neg",
         "fungal": "log10_um_fungal"}
TASK_CN = {"gram_pos": "抗革兰阳性 MIC", "gram_neg": "抗革兰阴性 MIC",
           "fungal": "抗真菌 MIC"}
BOND_CN = {"AMD": "酰胺键(头尾)", "DSB": "二硫键", "EST": "酯键", "TIE": "硫醚键",
           "DCB": "碳碳双键", "CAR": "碳碳键", "AMN": "胺键", "none": "无链内键",
           "p-XylB": "对二甲苯桥", "(E)-but-2-enyl-B": "烯烃交联(订书肽)",
           "CLK": "C-lock", "TRZB": "三唑桥", "ETH": "醚键", "TriMeBn": "三甲基苄",
           "BisMeBn": "双甲基苄", "DSeB": "二硒键", "but-2-ynyl-B": "炔烃交联(订书肽)",
           "IMN": "亚胺键"}


def log(m: str) -> None:
    print(m, flush=True)


def load_xy(task):
    """返回 (X, y, meta, feat_names)。X = 描述符 + 指纹。"""
    ds = pd.read_pickle(DDIR / "quant_dataset.pkl")
    z = np.load(DDIR / "quant_features.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    fn = [str(x) for x in z["feat_names"]]
    morgan, maccs = z["morgan"], z["maccs"]

    dcols = [f"d_{c}" for c in fn]
    missing = [c for c in dcols if c not in ds.columns]
    if missing:
        raise RuntimeError(f"数据集缺少描述符列（示例：{missing[:3]}）")

    ds = ds.set_index(ds["dbaasp_id"].astype(str))
    keep = [i for i in ids if i in ds.index]
    pos = {c: k for k, c in enumerate(ids)}
    o = [pos[c] for c in keep]
    X = np.hstack([ds.loc[keep, dcols].to_numpy(dtype=float),
                   morgan[o].astype(float), maccs[o].astype(float)])
    col = TASKS[task]
    y = ds.loc[keep, col].to_numpy(dtype=float)
    meta = ds.loc[keep, ["bond_primary", "fold_cluster", "cluster_id",
                         "n_mic_records", "n_species"]].reset_index()
    ok = np.isfinite(y)
    X, y, meta = X[ok], y[ok], meta[ok].reset_index(drop=True)

    # ── NaN 处理（必须记录，否则会静默影响结论）──
    #
    # 为什么会有 NaN：序列类描述符（charge / boman_index / instability /
    # aa_composition 展开等）只有在**有氨基酸序列**时才能算。
    # 本数据集里 DBAASP 来源的肽大部分只有 SMILES、没有序列，
    # 所以这些列天然为空。
    #
    # 影响：
    #   * LightGBM / XGBoost / RandomForest **能处理 NaN**（各自的分裂策略不同），
    #     直接喂原始 NaN 是可以的；
    #   * **Ridge 不能**（实测报 "Input X contains NaN" 直接崩溃）。
    #
    # 处理方式：统一用**列中位数填充**（比填 0 更稳，因为描述符量纲差异大），
    # 并记录被填充的比例，写进报告供论文交代。
    nan_frac = float(np.isnan(X).mean())
    nan_cols = int((np.isnan(X).sum(axis=0) > 0).sum())
    if nan_cols:
        col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_med, idx[1])
    log(f"    NaN 审计：{nan_cols} 列含缺失，缺失占比 {nan_frac*100:.2f}%"
        f" → 已用列中位数填充（供 Ridge 等不支持 NaN 的模型使用）")
    return X, y, meta, fn


def make_model(name, seed):
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             min_child_samples=5, subsample=0.9, subsample_freq=1,
                             colsample_bytree=0.3, reg_lambda=1.0,
                             random_state=seed, n_jobs=-1, verbose=-1)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                            min_child_weight=2, subsample=0.9, colsample_bytree=0.3,
                            reg_lambda=1.0, random_state=seed, n_jobs=-1,
                            verbosity=0, tree_method="hist")
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                     random_state=seed, n_jobs=1)
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=seed)
    if name == "mean":
        return None          # 特殊：均值基线
    raise ValueError(name)


def metrics(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ok = np.isfinite(p)
    if ok.sum() < 3:
        return None
    y_, p_ = y[ok], p[ok]
    err = p_ - y_
    return {"n": int(ok.sum()), "R2": float(r2_score(y_, p_)),
            "RMSE": float(np.sqrt(mean_squared_error(y_, p_))),
            "MAE": float(mean_absolute_error(y_, p_)),
            "within_1log": float((np.abs(err) <= 1.0).mean()),
            "within_0.5log": float((np.abs(err) <= 0.5).mean()),
            "bias": float(err.mean())}


def cv_oof(X, y, folds, name, seed):
    oof = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        if len(te) < 3 or len(tr) < 20:
            continue
        if name == "mean":
            oof[te] = float(np.mean(y[tr]))
            continue
        mdl = make_model(name, seed + int(f))
        try:
            mdl.fit(X[tr], y[tr])
            oof[te] = mdl.predict(X[te])
        except Exception:  # noqa: BLE001
            pass
    return oof


def self_test() -> int:
    log("=" * 70)
    log("  15_model_quant.py 自检")
    log("=" * 70)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    ip = DDIR / "quant_dataset.pkl"
    chk("定量数据集存在（需先跑 14 号脚本）", ip.exists(),
        "未找到" if not ip.exists() else f"{ip.stat().st_size//1024}KB")
    if not ip.exists():
        log("  ⚠ 数据集未就绪，仅做函数级自检")

    # 指标函数
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = metrics(y, y + 0.2)
    chk("完美+0.2 偏移 R²>0.9", m and m["R2"] > 0.9, f"{m and round(m['R2'],4)}")
    chk("within_1log=1", m and m["within_1log"] == 1.0)
    chk("bias 为正（+0.2）", m and abs(m["bias"] - 0.2) < 1e-9, f"{m and m['bias']}")
    m0 = metrics(y, np.full_like(y, y.mean()))
    chk("常数预测 R²≈0", m0 and abs(m0["R2"]) < 1e-9, f"{m0 and m0['R2']}")

    # 模型可构建
    for nm in ["lgbm", "xgb", "rf", "ridge"]:
        try:
            chk(f"模型可构建：{nm}", make_model(nm, 0) is not None)
        except Exception as e:  # noqa: BLE001
            chk(f"模型可构建：{nm}", False, str(e)[:50])
    chk("mean 基线返回 None（走特殊分支）", make_model("mean", 0) is None)

    # 端到端
    rng = np.random.RandomState(0)
    X = rng.randn(200, 20)
    yy = X[:, 0] * 1.5 + X[:, 1] * 0.5 + rng.randn(200) * 0.3
    folds = np.array([i % 5 for i in range(200)])
    for nm in ["lgbm", "ridge", "mean"]:
        oof = cv_oof(X, yy, folds, nm, 0)
        mm = metrics(yy, oof)
        chk(f"端到端 CV 可跑通：{nm}", mm is not None and mm["n"] > 100,
            str(mm and mm["n"]))
    oof = cv_oof(X, yy, folds, "ridge", 0)
    chk("ridge 在有信号数据上 R² 明显>0", metrics(yy, oof)["R2"] > 0.5,
        str(round(metrics(yy, oof)["R2"], 3)))
    oofm = cv_oof(X, yy, folds, "mean", 0)
    chk("mean 基线 R² 接近 0", abs(metrics(yy, oofm)["R2"]) < 0.3,
        str(round(metrics(yy, oofm)["R2"], 3)))

    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 70)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="新数据集上的定量模型训练与评估",
                                 epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", default=["gram_pos", "gram_neg", "fungal"],
                    choices=list(TASKS))
    ap.add_argument("--models", nargs="+",
                    default=["mean", "ridge", "lgbm", "xgb", "rf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--min-type-n", type=int, default=40,
                    help="分层评估时，类型样本少于此数则并入 OTHER")
    ap.add_argument("--holdout", action="store_true",
                    help="额外用固定 holdout 折（fold 0）评估")
    ap.add_argument("--save-models", action="store_true",
                    help="在全量数据上重训并保存模型（供 predict 用）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    rows, strat_rows, per_seed_rows = [], [], []
    saved = {}
    for task in args.task:
        X, y, meta, fn = load_xy(task)
        folds = meta["fold_cluster"].to_numpy()
        log("")
        log("=" * 96)
        log(f"  任务 {TASK_CN[task]}：n={len(y)}  特征 {X.shape[1]} 维  "
            f"折数={len(set(folds))}")
        log(f"    y 范围 {y.min():+.2f} ~ {y.max():+.2f}  中位 {np.median(y):+.2f}  "
            f"std {y.std():.2f}")
        log("=" * 96)

        oof_by_model = {}
        for mn in args.models:
            ps = []
            for sd in args.seeds:
                oof = cv_oof(X, y, folds, mn, sd)
                m = metrics(y, oof)
                if m:
                    ps.append(m)
                    per_seed_rows.append({"task": task, "model": mn, "seed": sd, **m})
            if not ps:
                continue
            agg = {"task": task, "model": mn, "n": ps[0]["n"], "n_seed": len(ps)}
            for k in ("R2", "RMSE", "MAE", "within_1log", "within_0.5log", "bias"):
                agg[f"{k}_mean"] = float(np.mean([p[k] for p in ps]))
                agg[f"{k}_std"] = (float(np.std([p[k] for p in ps], ddof=1))
                                   if len(ps) > 1 else 0.0)
            rows.append(agg)
            # 预测集成口径（多 seed 平均预测）
            oofs = np.vstack([cv_oof(X, y, folds, mn, sd) for sd in args.seeds])
            oof_by_model[mn] = np.nanmean(oofs, axis=0)
            mi = metrics(y, oof_by_model[mn])
            if mi:
                agg.update({f"ens_{k}": v for k, v in mi.items() if k != "n"})

        log("")
        log("  ① 各模型表现（per-seed 均值 ± 标准差）")
        log(f"    {'模型':<7}{'R²':>18}{'±1log':>16}{'RMSE':>16}{'集成R²':>10}")
        for r in [x for x in rows if x["task"] == task]:
            log(f"    {r['model']:<7}{r['R2_mean']:>+10.3f}±{r['R2_std']:.3f}"
                f"{r['within_1log_mean']:>10.3f}±{r['within_1log_std']:.3f}"
                f"{r['RMSE_mean']:>10.3f}±{r['RMSE_std']:.3f}"
                f"{r.get('ens_R2', float('nan')):>10.3f}")

        # 最优模型（排除 mean 基线）
        cand = [r for r in rows if r["task"] == task and r["model"] != "mean"]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["R2_mean"])
        bm = best["model"]
        base = next((r for r in rows if r["task"] == task and r["model"] == "mean"), None)
        log("")
        log(f"  → 最优模型：{bm}  R²={best['R2_mean']:+.3f}±{best['R2_std']:.3f}"
            + (f"   均值基线 R²={base['R2_mean']:+.3f}  "
               f"提升 {best['R2_mean']-base['R2_mean']:+.3f}" if base else ""))

        # ---- 分层：按环化类型 ----
        log("")
        log(f"  ② 按环化类型分层（模型={bm}，样本 <{args.min_type_n} 归入 OTHER）")
        bond = meta["bond_primary"].astype(str).to_numpy()
        vc = pd.Series(bond).value_counts()
        big = set(vc[vc >= args.min_type_n].index)
        bond2 = np.array([b if b in big else "OTHER" for b in bond])
        oof_b = oof_by_model[bm]
        log(f"    {'环化类型':<20}{'n':>6}{'R²':>10}{'±1log':>9}{'RMSE':>9}")
        for b, idx in pd.Series(range(len(bond2))).groupby(bond2):
            ii = idx.to_numpy()
            if len(ii) < 15:
                continue
            m = metrics(y[ii], oof_b[ii])
            if not m:
                continue
            strat_rows.append({"task": task, "model": bm, "bond": b, **m})
            log(f"    {BOND_CN.get(b, b):<20}{m['n']:>6}{m['R2']:>+10.3f}"
                f"{m['within_1log']:>9.3f}{m['RMSE']:>9.3f}")

        # ---- 可选：固定 holdout ----
        if args.holdout:
            te = folds == 0
            tr = ~te
            log("")
            log(f"  ③ 固定 holdout（fold 0 作测试：n_test={int(te.sum())}）")
            for mn in args.models:
                if mn == "mean":
                    p = np.full(int(te.sum()), float(np.mean(y[tr])))
                else:
                    mdl = make_model(mn, 42)
                    mdl.fit(X[tr], y[tr])
                    p = mdl.predict(X[te])
                m = metrics(y[te], p)
                if m:
                    log(f"    {mn:<7} R²={m['R2']:+.3f}  ±1log={m['within_1log']:.3f}  "
                        f"RMSE={m['RMSE']:.3f}  n={m['n']}")
                    strat_rows.append({"task": task, "model": mn, "bond": "HOLDOUT_fold0", **m})

        # ---- 保存最终模型 ----
        if args.save_models:
            mdl = make_model(bm, 42)
            mdl.fit(X, y)
            saved[f"{task}__{bm}"] = {"model": mdl, "task": task, "model_name": bm,
                                      "n_train": int(len(y)), "n_feat": int(X.shape[1]),
                                      "feat_names": fn, "y_mean": float(y.mean()),
                                      "y_std": float(y.std())}
            log(f"  → 已训练最终模型（全量 {len(y)} 条）用于预测")

    if rows:
        pd.DataFrame(rows).to_csv(OUTD / "quant_metrics.csv", index=False,
                                  encoding="utf-8-sig")
        log("")
        log(f"  ✅ {OUTD}/quant_metrics.csv ({len(rows)} 行)")
    if per_seed_rows:
        pd.DataFrame(per_seed_rows).to_csv(OUTD / "quant_per_seed.csv", index=False,
                                           encoding="utf-8-sig")
    if strat_rows:
        pd.DataFrame(strat_rows).to_csv(OUTD / "quant_by_bond.csv", index=False,
                                        encoding="utf-8-sig")
        log(f"  ✅ {OUTD}/quant_by_bond.csv ({len(strat_rows)} 行)")
    if saved:
        import pickle
        with open(OUTD / "quant_models.pkl", "wb") as fh:
            pickle.dump(saved, fh)
        log(f"  ✅ {OUTD}/quant_models.pkl（{len(saved)} 个模型，供预测使用）")
    # 最优模型汇总（排除 mean 基线）
    best_map = {}
    for t in args.task:
        cand = [r for r in rows if r["task"] == t and r["model"] != "mean"]
        if cand:
            b = max(cand, key=lambda r: r["R2_mean"])
            base = next((r for r in rows
                         if r["task"] == t and r["model"] == "mean"), None)
            best_map[t] = {
                "model": b["model"], "R2_mean": b["R2_mean"],
                "R2_std": b["R2_std"], "within_1log": b["within_1log_mean"],
                "n": b["n"],
                "mean_baseline_R2": base["R2_mean"] if base else None,
                "gain_over_baseline": (b["R2_mean"] - base["R2_mean"]) if base else None,
            }
    (OUTD / "quant_report.json").write_text(json.dumps(
        {"tasks": args.task, "models": args.models, "seeds": args.seeds,
         "n_rows": len(rows), "best": best_map,
         "notes": [
             "数据源为 DBAASP 全量拉取（4407 条肽 / 18 种环化类型），"
             "非旧论文B缓存（1504 条 / 基本仅二硫键）。",
             "特征 2277 维 = 62 维描述符 + 2048 Morgan + 167 MACCS。",
             "回归的 seed 抖动远大于分类（实测 R² std 可达 0.07），"
             "所有结论基于多 seed 均值±标准差。",
             "序列类描述符对无序列的肽为 NaN，已用列中位数填充；"
             "LightGBM/XGBoost 原本可直接处理 NaN。",
         ]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
