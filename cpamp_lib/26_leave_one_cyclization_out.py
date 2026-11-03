#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""26_leave_one_cyclization_out.py -- 留一环化类型泛化评估（核心指标）。

为什么这是核心
------------
用户的目标是"**在未覆盖的环化类型上也能泛化**"。
普通 CV 把同一环化类型的肽混在训练/测试两侧，测不出泛化能力。
必须用 **Leave-One-Cyclization-Type-Out (LOCTO)**：
    每次**整类**留出（如只留 AMD、只留 DSB、只留 TIE），
    在该类上评估 → 这才是"未见环化类型"的真实性能。

该框架还支持**特征组对比**：
    base     62 描述符 + 2215 指纹（现状）
    +exp     加入 DBAASP 实验环化注释
    +topo    加入新设计的**交联拓扑表征**（见 27 号脚本产出）

用法
----
    python 26_leave_one_cyclization_out.py --self-test
    python 26_leave_one_cyclization_out.py --strains "Pseudomonas aeruginosa ATCC 27853"
    python 26_leave_one_cyclization_out.py --all-target-strains
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
OUTD = ROOT / "artifacts" / "locto"
try:
    OUTD.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

#: 目标菌株（用户指定重点）
TARGET_STRAINS = [
    "Pseudomonas aeruginosa ATCC 27853",
    "Acinetobacter baumannii ATCC 19606",
    "Pseudomonas aeruginosa PAO1",
    "Acinetobacter baumannii ATCC BAA-1605",
]

BOND_CN = {"AMD": "酰胺键(头尾)", "DSB": "二硫键", "TIE": "硫醚键", "EST": "酯键",
           "DCB": "碳碳双键", "CAR": "碳碳键", "AMN": "胺键", "p-XylB": "对二甲苯桥",
           "(E)-but-2-enyl-B": "烯烃交联", "ETH": "醚键", "CLK": "C-lock",
           "TRZB": "三唑桥", "none": "无链内键"}


def log(m: str) -> None:
    print(m, flush=True)


def _load(mod_name: str, fname: str):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def build_features(topo_path: Path | None = None,
                   delta_path: Path | None = None,
                   exp_path: Path | None = None):
    """返回 [X_base, X_exp, X_topo, X_delta, keys, bonds, smis], EXP_COLS, TOPO_COLS, DELTA_COLS。

    ⚠️ `exp_cyclization.csv` 缺失时 `Xe` 为零宽（与 topo/delta 降级逻辑一致）：
       2026-10-18 清理事故中该文件被删，且无法从现有数据完整重建。
    """
    qd = pd.read_pickle(DDIR / "quant_dataset.pkl")
    z = np.load(DDIR / "quant_features.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    row = {c: i for i, c in enumerate(ids)}
    feat_cols = [c for c in qd.columns if c.startswith("d_")]

    # ⭐ exp 可降级
    _ep = exp_path if exp_path is not None else (DDIR / "exp_cyclization.csv")
    exp = None
    if _ep is not None and Path(_ep).exists():
        exp = pd.read_csv(_ep)
        exp["_sid"] = exp["dbaasp_id"].astype(str)
        exp = exp.drop_duplicates("_sid").set_index("_sid")
    # ⚠️ 坑：不能简单用 `startswith("exp")` —— 那会把字符串列 `exp_kind` 也抓进来，
    #    后面 .astype(float) 会崩（could not convert string to float）。
    #    只取数值型的 exp_n_* 与二值 exp_*。
    if exp is not None:
        num_cols = [c for c in exp.columns
                    if c.startswith("exp_n_") or c in
                    ("exp_has_ring", "exp_head_to_tail", "exp_side_chain",
                     "exp_mixed")]
        kind_dum = pd.get_dummies(exp["exp_kind"], prefix="kind").astype(float)
        exp_feat = pd.concat([exp[num_cols].astype(float), kind_dum], axis=1)
        EXP_COLS = list(exp_feat.columns)
    else:
        exp_feat = pd.DataFrame(index=[])
        EXP_COLS = []

    topo = None
    if topo_path is not None and topo_path.exists():
        topo = pd.read_csv(topo_path)
        topo["_sid"] = topo["dbaasp_id"].astype(str)
        topo = topo.drop_duplicates("_sid").set_index("_sid")
        TOPO_COLS = [c for c in topo.columns if c.startswith("topo_")]
    else:
        TOPO_COLS = []

    dl = None
    if delta_path is not None and delta_path.exists():
        dl = pd.read_csv(delta_path)
        dl["_sid"] = dl["dbaasp_id"].astype(str)
        dl = dl.drop_duplicates("_sid").set_index("_sid")
        DELTA_COLS = [c for c in dl.columns if c.startswith("dl_") and c != "dl_ok"]
    else:
        DELTA_COLS = []

    qd2 = qd.copy(); qd2["_sid"] = qd2["dbaasp_id"].astype(str)
    qd2 = qd2.drop_duplicates("_sid").set_index("_sid")

    Xb, Xe, Xt, Xd, keys, bonds, smis = [], [], [], [], [], [], []
    for c in ids:
        if c not in qd2.index:
            continue
        v = pd.to_numeric(qd2.loc[c, feat_cols], errors="coerce").astype(float).to_numpy()
        if v.shape[0] != len(feat_cols):
            continue
        Xb.append(np.concatenate([v, z["morgan"][row[c]].astype(float),
                                  z["maccs"][row[c]].astype(float)]))
        if EXP_COLS:
            Xe.append(exp_feat.loc[c].to_numpy(dtype=float)
                      if c in exp_feat.index else np.zeros(len(EXP_COLS)))
        if TOPO_COLS:
            Xt.append(topo.loc[c, TOPO_COLS].to_numpy(dtype=float) if c in topo.index
                      else np.zeros(len(TOPO_COLS)))
        if DELTA_COLS:
            Xd.append(dl.loc[c, DELTA_COLS].to_numpy(dtype=float) if c in dl.index
                      else np.zeros(len(DELTA_COLS)))
        keys.append(c)
        bonds.append(str(qd2.loc[c, "bond_primary"]) if "bond_primary" in qd2.columns else "")
        smis.append(str(qd2.loc[c, "smiles"]) if "smiles" in qd2.columns else "")
    out = [np.vstack(Xb),
           np.vstack(Xe) if EXP_COLS else np.zeros((len(keys), 0)),
           np.vstack(Xt) if TOPO_COLS else np.zeros((len(keys), 0)),
           np.vstack(Xd) if DELTA_COLS else np.zeros((len(keys), 0)),
           np.array(keys), np.array(bonds), smis]
    return out, EXP_COLS, TOPO_COLS, DELTA_COLS


def clean(X):
    X = X.copy()
    med = np.nanmedian(X, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    ii = np.where(np.isnan(X))
    if len(ii[0]):
        X[ii] = np.take(med, ii[1])
    return X


def self_test() -> int:
    log("=" * 72)
    log("  26_leave_one_cyclization_out.py 自检")
    log("=" * 72)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    # ⚠️ `exp_cyclization.csv` 已改为**可选**（2026-10-18 清理事故后可能缺失），
    #    build_features 会在缺失时降级为零宽 Xe。故不再列入必备清单。
    chk("数据文件齐备",
        all((DDIR / f).exists() for f in
            ("quant_dataset.pkl", "quant_features.npz", "mic_pairs.csv")))
    try:
        (Xb, Xe, Xt, Xd, keys, bonds, smis), ec, tc, dc = build_features()
        chk("基础特征 2277 维", Xb.shape[1] == 2277, str(Xb.shape))
        # ⭐ 降级容忍：exp 缺失时 Xe 为零宽，此时不视为失败
        if Xe.shape[1] > 0:
            chk("实验环化特征非空", True, f"{Xe.shape[1]} 维")
        else:
            chk("ⓘ 实验环化特征缺失 → 已降级为零宽（exp_cyclization.csv 不存在）",
                True)
        chk("样本数一致", Xb.shape[0] == len(keys) == len(bonds), str(len(keys)))
        chk("bond_primary 有多种取值", len(set(bonds)) >= 3,
            f"{len(set(bonds))} 种")
    except Exception as e:  # noqa: BLE001
        chk("特征构造可执行", False, f"{type(e).__name__}: {str(e)[:70]}")
    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 72)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="留一环化类型泛化评估")
    ap.add_argument("--strains", nargs="+", default=None)
    ap.add_argument("--all-target-strains", action="store_true")
    ap.add_argument("--topo", default=None, help="拓扑表征 CSV（27 号脚本产出）")
    ap.add_argument("--delta", default=None, help="Δ 表征 CSV（30 号脚本产出）")
    ap.add_argument("--min-train", type=int, default=150)
    ap.add_argument("--min-type", type=int, default=25, help="留出类型至少多少条")
    ap.add_argument("--models", nargs="+", default=["lgbm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    m15 = _load("m15", "15_model_quant.py")
    m13 = _load("m13", "13_model_by_cyclization.py")   # make_reg 在这里
    mp = pd.read_csv(DDIR / "mic_pairs.csv")
    topo_path = Path(args.topo) if args.topo else (DDIR / "topo_cyclization.csv")
    delta_path = getattr(args, "delta", None)
    delta_path = Path(delta_path) if delta_path else (DDIR / "delta_cyclization.csv")
    (Xb, Xe, Xt, Xd, keys, bonds, smis), EXP_COLS, TOPO_COLS, DELTA_COLS = \
        build_features(topo_path, delta_path)
    kidx = {k: i for i, k in enumerate(keys)}
    log(f"  特征：base {Xb.shape[1]} + exp {Xe.shape[1]} + topo {Xt.shape[1]} "
        f"+ delta {Xd.shape[1]}")

    if args.all_target_strains:
        species = TARGET_STRAINS
    elif args.strains:
        species = args.strains
    else:
        cnt = mp.groupby("species")["dbaasp_id"].nunique().sort_values(ascending=False)
        species = [s for s in cnt.index if cnt[s] >= 250][:3]

    feature_sets = [("base", lambda i: Xb[i])]
    if len(EXP_COLS):
        feature_sets.append(("base+exp", lambda i: np.hstack([Xb[i], Xe[i]])))
    if len(TOPO_COLS):
        feature_sets.append(("base+topo", lambda i: np.hstack([Xb[i], Xt[i]])))
    # ⭐ Δ 表征：环化导致的理化变化（跨类型可迁移的核心假设）
    if len(DELTA_COLS):
        feature_sets.append(("base+delta", lambda i: np.hstack([Xb[i], Xd[i]])))
        feature_sets.append(("base+delta+topo",
                             lambda i: np.hstack([Xb[i], Xd[i], Xt[i]])))
        if len(TOPO_COLS):
            feature_sets.append(("base+delta+topo+exp",
                                 lambda i: np.hstack([Xb[i], Xd[i], Xt[i], Xe[i]])))

    all_rows = []
    for sp in species:
        sub = mp[mp["species"] == sp]
        if not len(sub):
            continue
        per = sub.groupby("dbaasp_id")["log10_um"].median()
        per.index = per.index.astype(str)
        avail = [i for i in per.index if i in kidx]
        if len(avail) < args.min_train:
            log(f"  ⏭ {sp}: 可用 {len(avail)} 条，跳过")
            continue
        ii = np.array([kidx[i] for i in avail])
        y = per.reindex(avail).to_numpy(dtype=float)
        bd = bonds[ii]
        ok = np.isfinite(y)
        ii, y, bd = ii[ok], y[ok], bd[ok]

        # 找出样本量足够的环化类型
        vc = pd.Series(bd).value_counts()
        types = [t for t in vc.index if vc[t] >= args.min_type and t != "none"]
        log("")
        log("=" * 108)
        log(f"  【{sp}】n={len(y)}  可留出的环化类型: "
            f"{[(t, int(vc[t])) for t in types[:6]]}")
        log("=" * 108)

        for fs_name, fsfn in feature_sets:
            Xall = clean(fsfn(ii))
            # 基线：普通 5 折 CV（覆盖类型的性能）
            m14 = _load("m14", "14_build_quant_dataset.py")
            folds, _, _ = m14.cluster_split_balanced(
                [smis[j] for j in ii], threshold=0.7, n_folds=5, seed=42,
                labels=np.ones(len(y), bool))
            ps = [m15.metrics(y, m15.cv_oof(Xall, y, folds, mn, sd))
                  for mn in args.models for sd in args.seeds]
            ps = [p for p in ps if p]
            base_r2 = float(np.mean([p["R2"] for p in ps])) if ps else np.nan
            base_p1 = float(np.mean([p["within_1log"] for p in ps])) if ps else np.nan

            log("")
            log(f"  ── 特征组 {fs_name}（{Xall.shape[1]} 维）  普通CV: "
                f"R²={base_r2:+.3f}  ±1log={base_p1:.3f}")
            log(f"     {'留出类型':<22}{'训练n':>7}{'测试n':>7}{'R²':>10}{'±1log':>9}"
                f"{'RMSE':>9}  {'类型内均值基线':>14}")
            for t in types:
                te = bd == t
                tr = ~te
                if te.sum() < args.min_type or tr.sum() < args.min_train:
                    continue
                # 测试集内 y 方差决定 R² 上限
                yb = float(np.mean(y[tr]))
                base_m = m15.metrics(y[te], np.full(te.sum(), yb))
                res = []
                for mn in args.models:
                    for sd in args.seeds:
                        mdl = m13.make_reg(mn, sd)
                        try:
                            mdl.fit(Xall[tr], y[tr])
                            p = mdl.predict(Xall[te])
                        except Exception:  # noqa: BLE001
                            continue
                        mm = m15.metrics(y[te], p)
                        if mm:
                            res.append(mm)
                if not res:
                    continue
                r2 = float(np.mean([r["R2"] for r in res]))
                r2s = float(np.std([r["R2"] for r in res], ddof=1)) if len(res) > 1 else 0.0
                p1 = float(np.mean([r["within_1log"] for r in res]))
                rm = float(np.mean([r["RMSE"] for r in res]))
                log(f"     {BOND_CN.get(t, t):<22}{int(tr.sum()):>7}{int(te.sum()):>7}"
                    f"{r2:>+10.3f}{p1:>9.3f}{rm:>9.3f}  "
                    f"{base_m['R2'] if base_m else float('nan'):>+14.3f}")
                all_rows.append({"species": sp, "feature_set": fs_name,
                                 "held_out_type": t, "n_train": int(tr.sum()),
                                 "n_test": int(te.sum()), "y_test_std": float(y[te].std()),
                                 "R2": r2, "R2_std": r2s, "within_1log": p1,
                                 "RMSE": rm, "cv_R2_all_types": base_r2,
                                 "cv_within_1log_all_types": base_p1})

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTD / "locto_metrics.csv", index=False, encoding="utf-8-sig")
        log("")
        log("=" * 108)
        log("  汇总：留出类型 vs 普通 CV（泛化缺口）")
        log("=" * 108)
        for fs in df["feature_set"].unique():
            s = df[df["feature_set"] == fs]
            log(f"    [{fs}]  留出类型平均 R²={s['R2'].mean():+.3f}  "
                f"±1log={s['within_1log'].mean():.3f}  "
                f"（普通CV {s['cv_R2_all_types'].mean():+.3f}）  "
                f"缺口 {s['R2'].mean()-s['cv_R2_all_types'].mean():+.3f}")
        log("")
        log(f"  ✅ 已写出 {OUTD}/locto_metrics.csv（{len(df)} 行）")
    else:
        log("✗ 没有结果")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
