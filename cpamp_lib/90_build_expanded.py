#!/usr/bin/env python -X utf8
# -*- coding: utf-8 -*-
"""90_build_expanded.py -- 构建扩展数据集（多菌株 + 新骨架锚点）。

设计依据
-------
88/89 号盘点：
    · DBAASP 有 MIC 的唯一肽 **4407**，我们只用了 1451
    · 差集 **2956** 条，其中 **2188 条落在 1260 个新骨架上**（90.9%）
    · DBAASP 有 **2952 个菌株**，我们只用 6 个；**≥30 条的菌株有 159 个**

⇒ 两个扩展方向
    **A. 菌株扩展**（6 → 多个）：提升定量覆盖面与统计强度
    **B. 骨架扩展**（+1260 新骨架）：这才是**泛化的关键**（91 号已证冗余度 93.2%）

⚠️ 防泄漏设计（必须）
------------------
留出 4 个菌株作为**永久外部验证集**，**绝不进入训练**：
    Escherichia coli ATCC 25922 / Staphylococcus aureus ATCC 25923 /
    Pseudomonas aeruginosa PAO1 / Enterococcus faecalis ATCC 29212
（与 `artifacts/external_strain` 一致，保证外部数字可与历史对比）

产物
----
    `datasets/DBAASP_full/expanded.npz`     X + y + strain + ids + fold + names
    `datasets/DBAASP_full/expanded_meta.pkl` 元信息（菌株清单、统计）

用法
----
    python 90_build_expanded.py --self-test
    python 90_build_expanded.py --time            # 先测特征计算速度
    python 90_build_expanded.py --build --min-per-strain 30
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
OUTD = ROOT / "artifacts" / "expanded"
try:
    OUTD.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

#: ⭐ 永久外部验证菌株（绝不进训练）
HOLDOUT_STRAINS = {
    "Escherichia coli ATCC 25922",
    "Staphylococcus aureus ATCC 25923",
    "Pseudomonas aeruginosa PAO1",
    "Enterococcus faecalis ATCC 29212",
}


def log(m: str) -> None:
    print(m, flush=True)


def _load(mod_name: str, fname: str):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def self_test() -> int:
    log("=" * 78)
    log("  90_build_expanded.py 自检")
    log("=" * 78)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    for f, nm in [(DDIR / "mic_pairs.csv", "mic_pairs"),
                  (DDIR / "topo_cyclization.csv", "topo_csv"),
                  (DDIR / "delta_cyclization.csv", "delta_csv")]:
        chk(f"{nm} 存在", f.exists())
    # ⭐ 外部留出菌株必须与历史一致
    ex = ROOT / "artifacts" / "external_strain" / "_kd.pkl"
    if ex.exists():
        kd = pd.read_pickle(ex)
        got = set(kd["species"].astype(str).unique())
        chk("⭐ 留出菌株与历史外部验证集一致", got == HOLDOUT_STRAINS,
            f"{sorted(got)}")
    chk("ⓘ 留出菌株数 = 4", len(HOLDOUT_STRAINS) == 4)
    # 特征维度口径
    qd = pd.read_pickle(DDIR / "quant_dataset.pkl")
    dcols = [c for c in qd.columns if c.startswith("d_")]
    chk("描述符 62 维", len(dcols) == 62, str(len(dcols)))
    chk("⭐ 特征总数口径 62+2048+167+21+33 = 2331",
        62 + 2048 + 167 + 21 + 33 == 2331)
    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 78)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def compute_features(smiles_list, seq_list):
    """按与 33 号**完全一致**的口径复算 2331 维特征。"""
    m3 = _load("m3", "03_features.py")
    m26 = _load("m26", "26_leave_one_cyclization_out.py")
    m27 = _load("m27", "27_topo_cyclization_features.py")
    m30 = _load("m30", "30_delta_cyclization_features.py")
    qd = pd.read_pickle(DDIR / "quant_dataset.pkl")
    dcols = [c[2:] for c in qd.columns if c.startswith("d_")]

    work = pd.DataFrame({"CPID": [f"x{i}" for i in range(len(smiles_list))],
                         "smiles": smiles_list,
                         "sequence": seq_list,
                         "source": ["exp"] * len(smiles_list)})
    panel, fps, _ = m3.compute_rows(work, progress_every=0)
    pan = panel.set_index("CPID")
    fid = {str(c): i for i, c in enumerate(fps["ids"])}
    Xb, keep = [], []
    for i, c in enumerate(work["CPID"]):
        if c not in pan.index or c not in fid:
            continue
        v = (pd.to_numeric(pan.loc[c].reindex(dcols), errors="coerce")
             .astype(float).to_numpy())
        j = fid[c]
        arr = np.concatenate([v, fps["morgan"][j].astype(float),
                              fps["maccs"][j].astype(float)])
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        Xb.append(arr)
        keep.append(i)
    if not Xb:
        return None, None
    Xb = np.vstack(Xb).astype(np.float32)
    smi = [smiles_list[i] for i in keep]
    seq = [seq_list[i] for i in keep]
    dl = pd.DataFrame([m30.delta_features(s, q) for s, q in zip(smi, seq)])
    dl = dl[[c for c in dl.columns if c.startswith("dl_") and c != "dl_ok"]]
    tp = pd.DataFrame([m27.topo_features(s) for s in smi])
    tp = tp[[c for c in tp.columns if c.startswith("topo_")]]
    X = m26.clean(np.hstack([Xb, dl.to_numpy(dtype=np.float32),
                             tp.to_numpy(dtype=np.float32)]))
    assert X.shape[1] == 2331, f"维度 {X.shape[1]} != 2331"
    return X.astype(np.float32), np.array(keep)


def timeit(n=200):
    mp = pd.read_csv(DDIR / "mic_pairs.csv")
    mp["dbaasp_id"] = mp["dbaasp_id"].astype(str)
    mp["smiles"] = mp["smiles"].fillna("").astype(str)
    sub = mp.drop_duplicates("dbaasp_id").head(n)
    seqs = (mp.drop_duplicates("dbaasp_id").set_index("dbaasp_id")
            .reindex(sub["dbaasp_id"])["smiles"].tolist())
    t0 = time.time()
    X, keep = compute_features(sub["smiles"].tolist(), [""] * len(sub))
    dt = time.time() - t0
    log("")
    log(f"  ⏱ 特征计算速度：{len(sub)} 条用 {dt:.1f}s "
        f"⇒ **{dt/len(sub)*1000:.0f} ms/条**")
    log(f"     X {X.shape}  成功 {len(keep)}/{len(sub)}")
    for tot in [4500, 8000]:
        log(f"     ⇒ {tot} 条约需 **{dt/len(sub)*tot/60:.1f} 分钟**")
    return dt / len(sub)


def build(min_per_strain=30, exclude_holdout=True, sequence_mode="resolved_v2"):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    if sequence_mode not in ("legacy", "resolved_v2"):
        raise ValueError("sequence_mode must be legacy or resolved_v2")
    from cpamp_sequence import database_sequence_lookup, replace_sequence_block
    feature_dir = DDIR if sequence_mode == "legacy" else DDIR / "sequence_repair_v2"
    result_dir = OUTD if sequence_mode == "legacy" else ROOT / "artifacts" / "sequence_repair_v2"
    try:
        feature_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        result_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    t0 = time.time()
    mp = pd.read_csv(DDIR / "mic_pairs.csv")
    mp["dbaasp_id"] = mp["dbaasp_id"].astype(str)
    mp["species"] = mp["species"].astype(str)
    mp["smiles"] = mp["smiles"].fillna("").astype(str)
    mp["log10_um"] = pd.to_numeric(mp["log10_um"], errors="coerce")
    mp = mp[np.isfinite(mp["log10_um"]) & (mp["smiles"].str.len() > 10)]
    log("")
    log("=" * 104)
    log(f"  构建扩展数据集（每菌株 ≥{min_per_strain}，"
        f"{'排除' if exclude_holdout else '包含'} {len(HOLDOUT_STRAINS)} 个留出菌株）")
    log("=" * 104)

    g = mp.groupby("species")["dbaasp_id"].nunique()
    keep_sp = sorted(s for s in g[g >= min_per_strain].index)
    if exclude_holdout:
        keep_sp = [s for s in keep_sp if s not in HOLDOUT_STRAINS]
    log(f"  入选菌株 {len(keep_sp)} 个")
    sub = mp[mp["species"].isin(keep_sp)].copy()
    # 每 (肽,菌株) 取中位（同一菌株多次测定 → 中位聚合）
    ag = (sub.groupby(["dbaasp_id", "species"])
          .agg(y=("log10_um", "median"), n=("log10_um", "size"),
               smiles=("smiles", "first"), bond=("bond_primary", "first"))
          .reset_index())
    log(f"  (肽,菌株) 对 {len(ag)}   唯一肽 {ag['dbaasp_id'].nunique()}")

    # 特征（按唯一肽算一次）
    upids = list(dict.fromkeys(ag["dbaasp_id"]))
    umap = ag.drop_duplicates("dbaasp_id").set_index("dbaasp_id")
    smis = [umap.loc[p, "smiles"] for p in upids]
    log(f"  复算 {len(upids)} 条唯一肽的特征…")
    sequence_lookup = database_sequence_lookup() if sequence_mode != "legacy" else {}
    resolved_sequences = [sequence_lookup.get(pid, "") for pid in upids]
    X, keep = compute_features(smis, resolved_sequences)
    if sequence_mode == "resolved_v2":
        X = replace_sequence_block(X, [resolved_sequences[i] for i in keep])
    ok_ids = [upids[i] for i in keep]
    log(f"  ✅ 特征 {X.shape}（成功 {len(ok_ids)}/{len(upids)}）")
    idx = {p: i for i, p in enumerate(ok_ids)}
    ag = ag[ag["dbaasp_id"].isin(idx)].copy()
    ag["_xi"] = ag["dbaasp_id"].map(idx)
    log(f"  保留 (肽,菌株) 对 {len(ag)}")

    # 骨架（用于后续分层评估）
    scaf = []
    for p in ok_ids:
        m = Chem.MolFromSmiles(smis[upids.index(p)])
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else "")
        except Exception:  # noqa: BLE001
            scaf.append("")
    sarr = np.array(scaf)
    ag["_scaf"] = sarr[ag["_xi"].to_numpy()]

    # 分折：按肽聚类（同肽不跨折）
    rng = np.random.RandomState(42)
    folds = rng.randint(0, 5, size=len(ok_ids))
    fmap = dict(zip(ok_ids, folds))
    ag["fold"] = ag["dbaasp_id"].map(fmap)

    names = ["base_%d" % i for i in range(2277)] + \
        [c for c in pd.read_csv(DDIR / "delta_cyclization.csv").columns
         if c.startswith("dl_") and c != "dl_ok"] + \
        [c for c in pd.read_csv(DDIR / "topo_cyclization.csv").columns
         if c.startswith("topo_")]
    assert len(names) == 2331, f"列名数 {len(names)}"
    if sequence_mode == "resolved_v2":
        from cpamp_retrain import corrected_delta
        X, _ = corrected_delta(
            X, [smis[i] for i in keep], [resolved_sequences[i] for i in keep], names)

    np.savez_compressed(
        feature_dir / "expanded.npz",
        X=X, ids=np.array(ok_ids), strain=ag["species"].values,
        y=ag["y"].values, bond=ag["bond"].fillna("").values,
        fold=ag["fold"].values, scaf=ag["_scaf"].values,
        names=np.array(names, dtype=object))
    ag.to_pickle(feature_dir / "expanded_meta.pkl")

    # 统计
    log("")
    log("  ⭐ 扩展数据集统计")
    log(f"     (肽, 菌株) 对  : **{len(ag)}**")
    log(f"     唯一肽          : **{len(ok_ids)}**")
    log(f"     唯一骨架        : **{ag['_scaf'].nunique()}**")
    log(f"     菌株            : **{ag['species'].nunique()}**")
    log(f"     活性域          : {ag['y'].min():.2f} ~ {ag['y'].max():.2f}"
        f"（均值 {ag['y'].mean():.2f}）")
    old = 2332
    log(f"     ⭐ 相比原统一模型（{old} 对 / 1451 肽）: "
        f"**{len(ag)/old:.2f}×** 数据量")
    s = {"n_pairs": int(len(ag)), "n_pep": int(len(ok_ids)),
         "n_scaf": int(ag["_scaf"].nunique()),
         "n_strain": int(ag["species"].nunique()),
         "min_per_strain": int(min_per_strain),
         "exclude_holdout": bool(exclude_holdout),
         "holdout": sorted(HOLDOUT_STRAINS)}
    s["feature_version"] = sequence_mode
    s["feature_candidate"] = "fixed_delta" if sequence_mode == "resolved_v2" else "legacy"
    s["folding"] = "random_unique_peptide_5fold_not_structural_clustering"
    with open(result_dir / ("build.json" if sequence_mode == "legacy" else "expanded_rebuilt_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    log("")
    log(f"  ✅ {feature_dir}/expanded.npz   用时 {time.time()-t0:.0f}s")
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构建扩展数据集")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--min-per-strain", type=int, default=30)
    ap.add_argument("--sequence-mode", choices=["legacy", "resolved_v2"],
                    default="resolved_v2")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.time:
        timeit()
        return 0
    if args.build:
        build(min_per_strain=args.min_per_strain, sequence_mode=args.sequence_mode)
        return 0
    log("请指定 --build / --time / --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
