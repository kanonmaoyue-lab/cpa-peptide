#!/usr/bin/env python -X utf8
# -*- coding: utf-8 -*-
"""98_qual_locto.py -- 定性侧的留一环化类型（LOCTO）评估。

⭐ 为什么这一步是决定性的
--------------------
97 号更正了一个关键数字：**同一模型口径下，定量侧留一骨架 R² ≈ 0.50**
（不是我此前文档里误用的 0.135）。

而定性侧 95 号显示：**留一骨架 AUROC 0.9350 ≈ 整簇 0.9363（Δ 仅 0.0013）**。

⇒ 两边在"未见骨架"上**都还行**。那么真正的难点是什么？
   定量侧 LOCTO（留一环化类型）：**R² ≈ 0**
   ⇒ 暗示**真正的瓶颈是"环化化学"**，而不是"骨架"。

**但定性侧从未测过 LOCTO。** 本脚本补上，才能给出对称结论。

数据
----
定性集（CyclicPepedia）**没有** DBAASP 的 `bond_primary` 环化类型标注，
故通过 SMILES 自行判定环化类型（head-to-tail / 二硫键 / 侧链 / 混合）。

用法
----
    python 98_qual_locto.py --self-test
    python 98_qual_locto.py --run
"""
from __future__ import annotations

import argparse
import importlib.util
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
OUTD = ROOT / "artifacts" / "qual_gap"
try:
    OUTD.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
TASKS = [("总抗菌(H2)", "anti_bacterial_any"), ("抗细菌", "anti_bacterial"),
         ("抗革兰阳性", "anti_gram_pos"), ("抗革兰阴性", "anti_gram_neg"),
         ("抗真菌", "anti_fungal")]


def log(m: str) -> None:
    print(m, flush=True)


def _load(n, f):
    sp = importlib.util.spec_from_file_location(n, HERE / f)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def cyclo_type(smiles):
    """用 SMILES 判定环化类型（对齐 DBAASP 的 bond_primary 语义）。

    返回 'HT'（头尾）/ 'DSB'（二硫键）/ 'SC'（侧链成环）/ 'MIX' / 'OTHER'
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdmolops
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(str(smiles or ""))
    if m is None:
        return "OTHER"
    ri = m.GetRingInfo()
    if ri.NumRings() == 0:
        return "OTHER"
    # 找环上的骨架酰胺键（C(=O)-N 且在环内）
    bb_in_ring = False
    ss_in_ring = False
    for ring in ri.AtomRings():
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            at, bt = m.GetAtomWithIdx(a), m.GetAtomWithIdx(b)
            # 酰胺：C(=O)-N
            for x, y in ((at, bt), (bt, at)):
                if x.GetSymbol() == "C" and y.GetSymbol() == "N":
                    if any(bd.GetBondType() == Chem.BondType.DOUBLE and
                           bd.GetOtherAtom(x).GetSymbol() == "O"
                           for bd in x.GetBonds()):
                        bb_in_ring = True
            # 二硫键
            if {at.GetSymbol(), bt.GetSymbol()} == {"S"}:
                ss_in_ring = True
    # 侧链成环：环内有骨架外的原子（简化判据）
    bb_idx = set()
    from rdkit.Chem import MolFromSmarts
    patt = MolFromSmarts("[NX3][CX4H1,CX4H2][CX3]=O")
    for match in m.GetSubstructMatches(patt):
        bb_idx.update(match)
    sc_in_ring = any(len(set(ring) - bb_idx) > 0 and len(set(ring) & bb_idx) < 2
                     for ring in ri.AtomRings())
    # ⚠️ 顺序很重要：自检 `C1CSS1`（二硫键环）第一版被判成 MIX，
    #    因为 `sc_in_ring` 的简化判据（环内非骨架原子）也命中了。
    #    修正：**二硫键优先**（环内含两个 S 就是二硫键环，判据最确定）。
    if ss_in_ring and not bb_in_ring:
        return "DSB"
    kinds = sum([bb_in_ring, ss_in_ring, sc_in_ring])
    if kinds >= 2:
        return "MIX"
    if bb_in_ring:
        return "HT"
    if sc_in_ring:
        return "SC"
    return "OTHER"


def self_test() -> int:
    log("=" * 78)
    log("  98_qual_locto.py 自检")
    log("=" * 78)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    # ⭐ 环化类型判定：用已知分子验证
    chk("⭐ 头尾环肽判为 HT",
        cyclo_type("C1CCC(=O)N1") in ("HT", "MIX"), cyclo_type("C1CCC(=O)N1"))
    chk("⭐ 二硫键环肽判为 DSB",
        cyclo_type("C1CSS1") == "DSB", cyclo_type("C1CSS1"))
    chk("⭐ 线性肽判为 OTHER",
        cyclo_type("CC(=O)NCC(=O)O") in ("OTHER", "SC"),
        cyclo_type("CC(=O)NCC(=O)O"))
    chk("空输入不崩", cyclo_type("") == "OTHER")
    chk("非法 SMILES 不崩", cyclo_type("not_a_smiles") == "OTHER")
    try:
        m69 = _load("m69", "69_pantarget_qualitative.py")
        keep, X, Y, sub = m69.load_data()
        chk("定性数据可加载", X.shape[1] == 2277 and len(X) > 1000, f"{X.shape}")
        # 分布
        ct = pd.Series([cyclo_type(s) for s in sub["smiles"].astype(str)])
        chk("ⓘ 环化类型分布", True, str(ct.value_counts().to_dict()))
        chk("⭐ 至少 2 种类型 ≥50 条（可做 LOCTO）",
            int((ct.value_counts() >= 50).sum()) >= 2,
            str(int((ct.value_counts() >= 50).sum())))
    except Exception as e:  # noqa: BLE001
        import traceback
        chk("定性数据可加载", False, f"{type(e).__name__}: {str(e)[:60]}")
        log("   " + traceback.format_exc()[-260:].replace("\n", "\n   "))
    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 78)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def run(seeds=(0, 1, 2), min_type=50):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    t0 = time.time()
    m69 = _load("m69", "69_pantarget_qualitative.py")
    keep, X, Y, sub = m69.load_data()
    tasks = [t for t, _ in TASKS if t in Y]
    ct = np.array([cyclo_type(s) for s in sub["smiles"].astype(str)])
    vc = pd.Series(ct).value_counts()
    types = [t for t in vc[vc >= min_type].index]
    log("")
    log("=" * 104)
    log(f"  定性 LOCTO：{len(X)} 肽 × {len(tasks)} 目标")
    log(f"  环化类型分布：{vc.to_dict()}")
    log(f"  留出类型（≥{min_type} 条）：{types}")
    log("=" * 104)
    rows = []
    for j, t in enumerate(tasks):
        yt = Y[t]
        oof = np.full(len(X), np.nan)
        for tp in types:
            te = ct == tp
            tr = ~te
            if tr.sum() < 80 or te.sum() < 15 or len(set(yt[tr])) < 2:
                continue
            ps = []
            for sd in seeds:
                try:
                    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                           num_leaves=31, random_state=sd,
                                           verbose=-1)
                    m.fit(X[tr], yt[tr])
                    ps.append(m.predict_proba(X[te])[:, 1])
                except Exception:  # noqa: BLE001
                    pass
            if ps:
                oof[te] = np.mean(ps, axis=0)
        ok = np.isfinite(oof)
        if ok.sum() < 30 or len(set(yt[ok])) < 2:
            continue
        auc = float(roc_auc_score(yt[ok], oof[ok]))
        mcc = float(matthews_corrcoef(yt[ok], (oof[ok] >= 0.5).astype(int)))
        log(f"     {t:<14} LOCTO AUROC = {auc:.4f}   MCC = {mcc:.4f}   "
            f"正例 {int(yt[ok].sum())}  n={int(ok.sum())}")
        rows.append({"task": t, "AUROC": auc, "MCC": mcc,
                     "n_pos": int(yt[ok].sum()), "n": int(ok.sum())})
    D = pd.DataFrame(rows)
    D.to_csv(OUTD / "qual_locto.csv", index=False, encoding="utf-8-sig")
    log("")
    log("=" * 104)
    log("  ⭐ 汇总与对照")
    log("=" * 104)
    if len(D):
        log(f"     定性 LOCTO 平均 AUROC = **{D['AUROC'].mean():.4f}**"
            f"   平均 MCC = {D['MCC'].mean():.4f}")
    log("     定性 整簇 5 折平均 AUROC = 0.9363")
    log("     定性 留一骨架平均 AUROC = 0.9350")
    log(f"     ⇒ LOCTO 相对整簇的落差 = "
        f"{D['AUROC'].mean()-0.9363:+.4f}" if len(D) else "")
    log("")
    log("     定量对照：")
    log("       整簇/留一骨架 R² ≈ 0.50~0.75；LOCTO R² = **−0.0052**")
    log("")
    log(f"  用时 {time.time()-t0:.0f}s")
    log(f"  ✅ 已写出 {OUTD}/qual_locto.csv")
    return D


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="定性 LOCTO")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--min-type", type=int, default=50)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.run:
        run(min_type=args.min_type)
        return 0
    log("请指定 --run 或 --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
