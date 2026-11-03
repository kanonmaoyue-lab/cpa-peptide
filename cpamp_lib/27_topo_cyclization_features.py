#!/usr/bin/env python -X utf8
# -*- coding: utf-8 -*-
"""27_topo_cyclization_features.py -- 拓扑级环化表征（T1~T4）。

设计目标
-------
现有 `cyc_*` 特征能"识别"环化类型（区分准确率 0.933），但**无法外推**：
LOCTO（留一环化类型）实验显示留出类型 R² = −0.41，比均值基线还差。

根因：`cyc_*` 只描述"有几个环、环多大"，是**类型指纹**而非**物理量**。
本脚本改为编码**交联的化学与几何本质**，提供跨类型可迁移的信息：

  T1 交联键化学类型   用什么键成环（酰胺/二硫/硫醚/酯/碳碳/...）
  T2 交联参与链       环长在骨架还是侧链（主链-主链 / 侧链-侧链 / 混合）
  T3 环拓扑不变量     环大小分布、约束密度、交联间隔（可连续外推）
  T4 交联原子环境     成环原子是否芳香/杂原子邻接/取代度

**关键：全部从 SMILES 直接计算**，不依赖 DBAASP 注释——
这样训练数据与任意新肽（如芳基环烷肽）都能用同一套逻辑，表征是自洽的。

用法
----
    python 27_topo_cyclization_features.py --self-test
    python 27_topo_cyclization_features.py --input datasets/DBAASP_full/quant_dataset.pkl
    python 27_topo_cyclization_features.py --input 环肽清单.xlsx --out artifacts/topo_new.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DDIR = ROOT / "datasets" / "DBAASP_full"

#: T1 交联键化学类型（顺序固定，保证列名稳定）
#: ⚠️ 坑（实盘发现的 bug）：`"aromatic"` 必须在此列表里！
#:    `_ring_bond_kind()` 会返回 "aromatic"（环上全是芳香键），
#:    但下面 `d = {f"topo_n_{k}": 0.0 for k in BOND_KINDS}` 只初始化列表内的键。
#:    后果：**键集合随分子变化**——苯环分子返回 33 个键，其它分子只返回 32 个，
#:    下游按位置索引特征时会错位（预测脚本报 KeyError: 'topo_n_aromatic'）。
BOND_KINDS = ["amide", "disulfide", "thioether", "ester", "ether", "amine",
              "imine", "cc_single", "cc_double", "cc_triple", "aromatic",
              "aromatic_link", "aryl_alkyl", "unknown"]
#: T2 参与链类型
CHAIN_KINDS = ["mm", "ss", "sm"]

#: 主链原子标记：肽键的 N、Cα、羰基 C
BACKBONE_PAT = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)")


def log(m: str) -> None:
    print(m, flush=True)


def _bb_atoms(mol):
    """返回属于"主链"的原子索引集合（N-Cα-C(=O) 三原子组）。"""
    out = set()
    if BACKBONE_PAT is None:
        return out
    for match in mol.GetSubstructMatches(BACKBONE_PAT):
        out.update(match)
    return out


def _ring_bond_kind(mol, ring):
    """判断一个环主要由什么键合方式形成（T1）。

    做法：看环上**非骨架碳-碳单键**的键型；优先识别特征官能团。
    """
    idx = set(ring)
    has = {"disulfide": False, "amide": False, "ester": False, "ether": False,
           "thioether": False, "amine": False, "imine": False,
           "cc_double": False, "cc_triple": False, "aromatic": False,
           "aryl_alkyl": False, "cc_single": False}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in idx or j not in idx:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        z1, z2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        zs = {z1, z2}
        bt = b.GetBondType()
        if a1.GetIsAromatic() and a2.GetIsAromatic():
            has["aromatic"] = True
        if zs == {16}:
            has["disulfide"] = True
        elif zs == {6, 7} and bt == Chem.BondType.SINGLE:
            has["amine"] = True
        elif zs == {6, 7} and bt == Chem.BondType.DOUBLE:
            has["imine"] = True
        elif zs == {6, 8} and bt == Chem.BondType.SINGLE:
            # ⚠️ 坑：酯的"单键氧"也连在羰基碳上，只看键级会误判为醚。
            #    判据：该单键氧是否连着一个羰基碳（C=O）→ 是则为酯。
            oa = a1 if a1.GetAtomicNum() == 8 else a2
            ca = a2 if a1.GetAtomicNum() == 8 else a1
            is_ester_o = any(
                x.GetBondTypeAsDouble() == 2
                and x.GetOtherAtom(ca).GetAtomicNum() == 8
                for x in ca.GetBonds()
            )
            if is_ester_o:
                has["ester"] = True
            else:
                has["ether"] = True
        elif zs == {6, 8} and bt == Chem.BondType.DOUBLE:
            has["ester"] = True
        elif zs == {6, 16}:
            has["thioether"] = True
        elif zs == {6}:
            if bt == Chem.BondType.DOUBLE:
                has["cc_double"] = True
            elif bt == Chem.BondType.TRIPLE:
                has["cc_triple"] = True
            elif a1.GetIsAromatic() != a2.GetIsAromatic():
                has["aryl_alkyl"] = True
            elif not (a1.GetIsAromatic() and a2.GetIsAromatic()):
                has["cc_single"] = True
    # 酰胺：环上有 C(=O)-N
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            for ca, na in ((a1, a2), (a2, a1)):
                if ca.GetAtomicNum() == 6 and na.GetAtomicNum() == 7:
                    if any(x.GetBondTypeAsDouble() == 2 and x.GetOtherAtom(ca).GetAtomicNum() == 8
                           for x in ca.GetBonds()):
                        has["amide"] = True
    # 优先级：特征官能团 > 芳环 > 普通碳碳
    for k in ["disulfide", "amide", "ester", "thioether", "ether", "imine",
              "amine", "cc_triple", "cc_double", "aryl_alkyl", "aromatic",
              "cc_single"]:
        if has[k]:
            return k
    return "unknown"


def _chain_kind(mol, ring, bb, bond_kind="unknown"):
    """T2：环主要连在主链还是侧链。

    ⚠️ 重要修正（实测发现的 bug）：
    早先只按"环内主链原子占比"判定，结果**纯二硫键肽被误判为主链环**
    （实测 chain_mm 均值 1.65，而它应为 0）。
    原因是二硫键环含 2 个 Cα，而 Cα 被我标为主链原子 → 占比 0.67 > 0.6。

    正确做法：**按交联键的化学本质决定归类**——
    二硫键、硫醚、酯、醚、胺、亚胺这类键**天生是侧链交联**
    （它们连在侧链官能团上），不论环里含多少主链原子。
    只有酰胺键/碳碳键才需要按主链占比区分头尾 vs 侧链。
    """
    if bond_kind in ("disulfide", "thioether", "ester", "ether", "amine",
                     "imine", "cc_double", "cc_triple", "aryl_alkyl"):
        return "ss"
    # 酰胺键 / 芳香 / 普通碳碳：看主链原子占比
    n_bb = sum(1 for i in ring if i in bb)
    frac = n_bb / max(len(ring), 1)
    if frac >= 0.6:
        return "mm"
    if frac <= 0.2:
        return "ss"
    return "sm"


def topo_features(smiles: str) -> dict:
    """计算 T1~T4 全部特征。返回 dict（键以 topo_ 开头）。"""
    d = {f"topo_n_{k}": 0.0 for k in BOND_KINDS}
    d.update({f"topo_chain_{k}": 0.0 for k in CHAIN_KINDS})
    base = {
        "topo_ok": 0.0, "topo_n_rings": 0.0, "topo_n_macro": 0.0,
        "topo_max_ring": 0.0, "topo_ring_mean": 0.0, "topo_ring_std": 0.0,
        "topo_constraint_density": 0.0, "topo_frac_atoms_in_rings": 0.0,
        "topo_mean_linker_dist": 0.0, "topo_max_linker_dist": 0.0,
        "topo_frac_aromatic_in_ring": 0.0, "topo_frac_hetero_in_ring": 0.0,
        "topo_frac_substituted_ring_atoms": 0.0, "topo_ring_size_entropy": 0.0,
        "topo_n_ss_bonds": 0.0, "topo_amide_in_ring": 0.0,
    }
    d.update(base)

    mol = Chem.MolFromSmiles(str(smiles)) if smiles and str(smiles) != "nan" else None
    if mol is None:
        return d
    d["topo_ok"] = 1.0
    n_atoms = mol.GetNumAtoms()
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if not rings:
        return d
    bb = _bb_atoms(mol)

    # 环尺寸统计
    sizes = np.array([len(r) for r in rings], dtype=float)
    macro = sizes >= 12          # 大环（≥12 原子；二硫键环通常 5-6，主链环 ≥9）
    d["topo_n_rings"] = float(len(rings))
    d["topo_n_macro"] = float(macro.sum())
    d["topo_max_ring"] = float(sizes.max())
    d["topo_ring_mean"] = float(sizes.mean())
    d["topo_ring_std"] = float(sizes.std())
    # 环大小熵（多样性）
    if len(sizes) > 1:
        _, cnt = np.unique(sizes, return_counts=True)
        p = cnt / cnt.sum()
        d["topo_ring_size_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
    # 约束密度：环原子 / 总原子
    allring = set().union(*rings)
    d["topo_constraint_density"] = float(len(allring) / max(n_atoms, 1))
    d["topo_frac_atoms_in_rings"] = d["topo_constraint_density"]

    # 只看"大环"（真正决定环肽性质的交联）
    big = [r for r in rings if len(r) >= 12] or rings
    for r in big:
        k = _ring_bond_kind(mol, r)
        d[f"topo_n_{k}"] = d.get(f"topo_n_{k}", 0.0) + 1.0
        c = _chain_kind(mol, r, bb, k)
        d[f"topo_chain_{c}"] += 1.0
    # 二硫键总数（全分子，不限大环）
    d["topo_n_ss_bonds"] = float(sum(
        1 for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() == 16
        and b.GetEndAtom().GetAtomicNum() == 16))
    d["topo_amide_in_ring"] = d.get("topo_n_amide", 0.0)

    # T3 交联间隔：环上"跨环原子对"的最短路径近似 = 环长/2（用环尺寸分布代替）
    # 更实用的是：不同环之间共享原子数（稠合程度）
    if len(rings) >= 2:
        shared = []
        for a in range(len(rings)):
            for b_ in range(a + 1, len(rings)):
                shared.append(len(rings[a] & rings[b_]))
        d["topo_mean_linker_dist"] = float(np.mean(shared))
        d["topo_max_linker_dist"] = float(np.max(shared))

    # T4 环上原子环境
    ar = [1.0 if mol.GetAtomWithIdx(i).GetIsAromatic() else 0.0 for i in allring]
    het = [1.0 if mol.GetAtomWithIdx(i).GetAtomicNum() not in (6,) else 0.0
           for i in allring]
    sub = []
    for i in allring:
        a = mol.GetAtomWithIdx(i)
        deg_in_ring = sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in allring)
        sub.append(1.0 if a.GetDegree() > deg_in_ring else 0.0)
    d["topo_frac_aromatic_in_ring"] = float(np.mean(ar)) if ar else 0.0
    d["topo_frac_hetero_in_ring"] = float(np.mean(het)) if het else 0.0
    d["topo_frac_substituted_ring_atoms"] = float(np.mean(sub)) if sub else 0.0
    return d


TOPO_PREFIX = "topo_"


def self_test() -> int:
    log("=" * 74)
    log("  27_topo_cyclization_features.py 自检")
    log("=" * 74)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    # 1) 线性分子：无环
    d = topo_features("CCCCCC")
    chk("线性肽：n_rings=0", d["topo_n_rings"] == 0, str(d["topo_n_rings"]))
    chk("线性肽：ok=1（能解析）", d["topo_ok"] == 1.0)

    # 2) 二硫键环（8 元环）：S-S 形成
    d = topo_features("C1CCSSCC1")
    chk("含二硫键环：识别到 disulfide",
        d["topo_n_disulfide"] >= 1, str(d["topo_n_disulfide"]))
    chk("含二硫键环：n_ss_bonds>=1", d["topo_n_ss_bonds"] >= 1)

    # 3) 环己烷：纯碳环
    d = topo_features("C1CCCCC1")
    chk("环己烷：识别到 cc_single", d["topo_n_cc_single"] >= 1,
        str(d["topo_n_cc_single"]))
    chk("环己烷：约束密度=1.0", abs(d["topo_constraint_density"] - 1.0) < 1e-9)

    # 4) 苯环：芳香
    d = topo_features("c1ccccc1")
    chk("苯环：识别到 aromatic", d["topo_n_aromatic"] >= 1, str(d["topo_n_aromatic"]))
    chk("苯环：芳香占比=1.0", abs(d["topo_frac_aromatic_in_ring"] - 1.0) < 1e-9)

    # 5) 内酯环（酯键）
    d = topo_features("O=C1CCCO1")
    chk("内酯：识别到 ester", d["topo_n_ester"] >= 1, str(d["topo_n_ester"]))

    # 6) 内酰胺环（酰胺键）
    d = topo_features("O=C1CCCCN1")
    chk("内酰胺：识别到 amide", d["topo_n_amide"] >= 1, str(d["topo_n_amide"]))

    # 7) 真实头尾环化肽 Gramicidin S（项目数据库 CP00038）
    try:
        ds = pd.read_pickle(DDIR / "quant_dataset.pkl")
    except Exception:  # noqa: BLE001
        ds = None
    if ds is not None and "smiles" in ds.columns:
        # 找一条含二硫键的
        for s in ds["smiles"].head(400):
            d = topo_features(str(s))
            if d["topo_n_ss_bonds"] > 0:
                chk("真实二硫键肽：识别到 S-S", d["topo_n_ss_bonds"] > 0)
                break
        # 特征维度稳定
        d1 = topo_features("C1CCCCC1"); d2 = topo_features("CCO")
        chk("特征键集合稳定（同维度）", set(d1) == set(d2),
            f"{len(set(d1) ^ set(d2))} 个差异")

    # 8) 非法 SMILES 不崩
    d = topo_features("这不是SMILES")
    chk("非法 SMILES：ok=0 且不抛异常", d["topo_ok"] == 0.0)
    chk("非法 SMILES：仍返回完整键", f"topo_n_rings" in d)

    # 9) 芳基环烷桥（模拟那 13 条肽的关键基序：芳环 + 烷基链成环）
    d = topo_features("c1ccc2ccccc2c1")   # 萘
    chk("多环芳烃：n_rings>=2", d["topo_n_rings"] >= 2, str(d["topo_n_rings"]))

    # 10) ⭐ 结构一致性守卫：**任意**分子的键集合必须完全相同
    #     这一项专门防住"键随分子变化"这类隐蔽 bug（曾导致下游按位置索引错位）
    probes = {
        "线性烷烃": "CCCCCC",
        "纯苯环": "c1ccccc1",
        "环己烷": "C1CCCCC1",
        "二硫键环": "C1CCSSCC1",
        "内酯": "O=C1CCCO1",
        "内酰胺": "O=C1CCCCN1",
        "萘": "c1ccc2ccccc2c1",
        "无环": "CCO",
        "非法": "这不是SMILES",
    }
    ref_keys = set(topo_features("CCCCCC"))
    bad = {k: len(set(topo_features(v)) ^ ref_keys) for k, v in probes.items()}
    bad = {k: v for k, v in bad.items() if v != 0}
    chk("所有分子的拓扑特征键集合一致（防键漂移）", not bad,
        f"键数={len(ref_keys)}；不一致的分子={bad}")
    # 键数 = BOND_KINDS 的 topo_n_* + CHAIN_KINDS 的 topo_chain_* + 其它固定键
    # 注意：`topo_n_*` 里还含 3 个非键型的固定键
    #       （topo_n_rings / topo_n_macro / topo_n_ss_bonds），故用包含关系而非等号。
    n_bond_prefix = sum(1 for k in ref_keys if k.startswith("topo_n_"))
    fixed_n = {"topo_n_rings", "topo_n_macro", "topo_n_ss_bonds"}
    chk("BOND_KINDS 全部落在 topo_n_* 键中",
        set(BOND_KINDS) <= {k[len("topo_n_"):] for k in ref_keys
                            if k.startswith("topo_n_")},
        str(sorted(set(BOND_KINDS) - {k[len("topo_n_"):] for k in ref_keys
                                      if k.startswith("topo_n_")})))
    chk("topo_n_* 键数 = 键型数 + 固定键数",
        n_bond_prefix == len(BOND_KINDS) + len(fixed_n),
        f"{n_bond_prefix} vs {len(BOND_KINDS)}+{len(fixed_n)}")
    n_chain = sum(1 for k in ref_keys if k.startswith("topo_chain_"))
    chk("CHAIN_KINDS 项数 == topo_chain_* 键数", n_chain == len(CHAIN_KINDS),
        f"{n_chain} vs {len(CHAIN_KINDS)}")
    # 每个 BOND_KINDS 都必须有对应键
    missing_kinds = [k for k in BOND_KINDS if f"topo_n_{k}" not in ref_keys]
    chk("BOND_KINDS 每项都有对应特征键", not missing_kinds, str(missing_kinds))

    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 74)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="拓扑级环化表征 T1~T4",
                                 epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DDIR / "quant_dataset.pkl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    p = Path(args.input)
    if not p.exists():
        log(f"✗ 找不到 {p}")
        return 2
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_pickle(p)

    low = {c.lower(): c for c in df.columns}
    idc = low.get("id") or low.get("dbaasp_id") or low.get("cpid") or df.columns[0]
    smic = low.get("smiles")
    if smic is None:
        log("✗ 缺少 SMILES 列")
        return 2
    log(f"  读入 {len(df)} 条（ID='{idc}', SMILES='{smic}'）")

    rows = []
    for _, r in df.iterrows():
        f = topo_features(str(r[smic]))
        f["dbaasp_id"] = r[idc]
        rows.append(f)
    out = pd.DataFrame(rows)
    cols = ["dbaasp_id"] + [c for c in out.columns if c.startswith(TOPO_PREFIX)]
    out = out[cols]
    log(f"  产出 {out.shape[1]-1} 个拓扑特征")
    log(f"  可解析: {int(out['topo_ok'].sum())}/{len(out)}")

    # 特征统计
    log("")
    log("=" * 96)
    log("  拓扑特征分布（非零方差才有信息）")
    log("=" * 96)
    log(f"  {'特征':<38}{'均值':>10}{'std':>9}{'非零数':>8}")
    for c in cols[1:]:
        v = pd.to_numeric(out[c], errors="coerce")
        if v.std() > 0:
            log(f"  {c:<38}{v.mean():>10.4f}{v.std():>9.4f}{int((v!=0).sum()):>8}")
    zero_cols = [c for c in cols[1:] if pd.to_numeric(out[c], errors="coerce").std() == 0]
    if zero_cols:
        log("")
        log(f"  ⚠ 常量特征（{len(zero_cols)} 个，建模时可删）: {zero_cols}")

    outp = Path(args.out) if args.out else (DDIR / "topo_cyclization.csv")
    out.to_csv(outp, index=False, encoding="utf-8-sig")
    log("")
    log(f"  ✅ 已写出 {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
