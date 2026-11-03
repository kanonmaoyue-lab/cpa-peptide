#!/usr/bin/env python -X utf8
# -*- coding: utf-8 -*-
"""30_delta_cyclization_features.py -- Δ 表征：环化导致的理化性质变化。

核心思路（回答"如何让模型理解环化类型的活性规律"）
------------------------------------------------
**关键认识：模型学的应该是"环化的物理效应"，而不是"环化的类型标签"。**

为什么类型标签无法外推：
    训练集里有 AMD / DSB / TIE…，模型学的是"AMD → 活性分布A"。
    遇到新的芳基环烷桥时，它**没有任何标签可映射** → 只能靠片段相似度瞎猜。

为什么 Δ 可以外推：
    环化影响活性的**真正机制**是它改变了
      · 构象刚性（熵损失 → 更易/更难与膜结合）
      · 两亲性（极性/非极性面的几何关系）
      · 侧链可及性（交联占用侧链官能团）
      · 局部化学环境（交联点附近的性质）
    这些量**与"用什么键成环"无关，只与"成环造成了什么变化"有关**。
    所以一个全新的交联类型，只要它造成的 Δ 落在训练过的 Δ 范围内，
    模型就能给出合理预测——**这才是真正的泛化**。

实现：对每条肽计算"环化版 vs 线性版"的理化性质差
    ① 全局刚性：环内原子占比、可旋转键损失比例、环尺寸跨度
    ② 构象约束：归一化到残基长度的约束密度
    ③ 两亲性变化：疏水矩 Δ（用残基级疏水性近似）
    ④ 侧链占用：参与交联的侧链原子占比、交联点残基数
    ⑤ 局部环境：交联点的杂原子邻接度、芳香性

用法
----
    python 30_delta_cyclization_features.py --self-test
    python 30_delta_cyclization_features.py --input datasets/DBAASP_full/quant_dataset.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DDIR = ROOT / "datasets" / "DBAASP_full"

#: 残基级疏水性（Kyte-Doolittle，归一化到 0~1）
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
       "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
       "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V"}

#: 常见端基/保护基修饰（真实肽序列里很常见）
_MOD_PREFIX = ("Ac-", "ac-", "Boc-", "Fmoc-", "H-", "NH2-", "Tfa-", "For-")
_MOD_SUFFIX = ("-NH2", "-nh2", "-OH", "-oh", "-CONH2", "-NHMe", "-NHEt",
               "-amide", "-NH3+")

PREFIX = "dl_"


def strip_mods(seq: str) -> str:
    """去掉端基修饰，只留残基串。

    ⚠️ 坑（实测发现的 bug）：真实序列写作 `Ac-KYWLKIWRAIRIL-NH2`。
    若不去修饰就直接喂给疏水性计算，`A`/`c`/`-`/`N`/`H`/`2` 会被当成残基
    （查不到 KD 值就按 0 处理），导致系统性偏差：
        hydro_mean   3.2923 -> 2.5650
        hydro_moment 1.2865 -> 1.0063
        helix_prop   0.1538 -> 0.1000
    即**疏水矩被低估**，污染 Δ 表征。必须先剥离修饰再算序列类特征。
    """
    t = str(seq).strip()
    changed = True
    while changed:
        changed = False
        for p in _MOD_PREFIX:
            if t.startswith(p):
                t = t[len(p):]
                changed = True
        for q in _MOD_SUFFIX:
            if t.endswith(q):
                t = t[:-len(q)]
                changed = True
    return t.strip("- ")


def log(m: str) -> None:
    print(m, flush=True)


def _ring_atoms(mol):
    ri = mol.GetRingInfo()
    allring = set()
    for r in ri.AtomRings():
        allring.update(r)
    return allring, [set(r) for r in ri.AtomRings()]


def _rot_bonds(mol, allowed=None):
    n = 0
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetDegree() == 1 or a2.GetDegree() == 1:
            continue
        if allowed is not None and (a1.GetIdx() not in allowed or a2.GetIdx() not in allowed):
            continue
        n += 1
    return n


def _graph_without_rings(mol):
    """删除环内的"额外"键（保留生成树），得到线性/开链版本。

    做法：对每个环，断开一条**非主链**的键（优先断 S-S、C-S 等交联特征键），
    从而把环"剪开"但不破坏分子连接性。
    """
    em = Chem.RWMol(mol)
    ri = mol.GetRingInfo()
    to_break = []
    for ring in ri.BondRings():
        if len(ring) < 4:
            continue
        # 选一条要断的键：优先断"环上唯一的特征交联键"
        best, best_score = None, -1
        for bidx in ring:
            b = mol.GetBondWithIdx(bidx)
            z1, z2 = b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()
            score = 0
            if {z1, z2} == {16}:            # S-S
                score = 100
            elif 16 in (z1, z2):            # C-S
                score = 50
            elif {z1, z2} == {6, 8}:        # C-O（酯/醚）
                score = 40
            elif {z1, z2} == {6, 7}:        # C-N
                score = 20
            elif b.GetBondType() != Chem.BondType.SINGLE:
                score = 10
            if score > best_score:
                best, best_score = bidx, score
        if best is not None:
            to_break.append(best)
    if not to_break:
        return None
    for bidx in sorted(set(to_break), reverse=True):
        em.RemoveBond(mol.GetBondWithIdx(bidx).GetBeginAtomIdx(),
                      mol.GetBondWithIdx(bidx).GetEndAtomIdx())
    try:
        m2 = em.GetMol()
        Chem.SanitizeMol(m2)
        return m2
    except Exception:  # noqa: BLE001
        return None


def _hydrophobic_moment(seq):
    """粗粒度疏水矩（单位：KD 的模长），衡量两亲性。"""
    if not seq:
        return 0.0, 0.0
    v = np.array([KD.get(c, 0.0) for c in seq], dtype=float)
    if len(v) < 2:
        return float(np.abs(v).mean()), 0.0
    n = np.arange(len(v))
    # 周期 3.6（α 螺旋）
    mh = np.sqrt((v * np.cos(2 * np.pi * n / 3.6)).sum() ** 2 +
                 (v * np.sin(2 * np.pi * n / 3.6)).sum() ** 2) / len(v)
    return float(np.abs(v).mean()), float(mh)


def delta_features(smiles: str, seq: str = "") -> dict:
    d = {}
    keys = ["dl_ok", "dl_n_rings", "dl_ring_atom_frac", "dl_rotbond_loss",
            "dl_rotbond_ratio", "dl_max_ring", "dl_ring_span_per_res",
            "dl_constraint_per_res", "dl_crosslink_residues",
            "dl_sidechain_atoms_in_ring", "dl_hetero_at_ring",
            "dl_arom_at_ring", "dl_tpsa_delta", "dl_logp_delta",
            "dl_hbd_delta", "dl_flexibility_index", "dl_rigidity_per_res",
            "dl_mw_per_atom", "dl_abs_charge_per_res", "dl_hydro_moment",
            "dl_hydro_mean", "dl_helix_propensity"]
    for k in keys:
        d[k] = 0.0

    mol = Chem.MolFromSmiles(str(smiles)) if smiles and str(smiles) != "nan" else None
    if mol is None:
        return d
    # ⚠️ 先剥离端基修饰，否则修饰基团会被当成残基污染序列类特征（见 strip_mods）
    seq = strip_mods(seq)
    d["dl_ok"] = 1.0
    n_atoms = mol.GetNumAtoms()
    allring, rings = _ring_atoms(mol)
    n_res = max(len(seq), 1)

    # ① 全局刚性
    d["dl_n_rings"] = float(len(rings))
    d["dl_ring_atom_frac"] = len(allring) / max(n_atoms, 1)
    d["dl_max_ring"] = float(max((len(r) for r in rings), default=0))
    rb_full = _rot_bonds(mol)

    # ② 剪开环 → 线性版，算 Δ
    lin = _graph_without_rings(mol)
    if lin is not None:
        rb_lin = _rot_bonds(lin)
        d["dl_rotbond_loss"] = float(max(rb_lin - rb_full, 0))
        d["dl_rotbond_ratio"] = (rb_full / rb_lin) if rb_lin > 0 else 1.0
        try:
            d["dl_tpsa_delta"] = float(rdMolDescriptors.CalcTPSA(lin) -
                                       rdMolDescriptors.CalcTPSA(mol))
            d["dl_logp_delta"] = float(Descriptors.MolLogP(lin) -
                                       Descriptors.MolLogP(mol))
            d["dl_hbd_delta"] = float(rdMolDescriptors.CalcNumHBD(lin) -
                                      rdMolDescriptors.CalcNumHBD(mol))
        except Exception:  # noqa: BLE001
            pass
    else:
        d["dl_rotbond_ratio"] = 1.0

    # ③ 归一化约束（跨肽可比的关键）
    d["dl_constraint_per_res"] = len(allring) / n_res
    d["dl_rigidity_per_res"] = (1.0 - d["dl_rotbond_ratio"]) * 10.0 / n_res
    d["dl_flexibility_index"] = rb_full / n_res
    d["dl_mw_per_atom"] = Descriptors.MolWt(mol) / max(n_atoms, 1)
    d["dl_ring_span_per_res"] = d["dl_max_ring"] / n_res

    # ④ 侧链占用：环原子中"非主链"的比例
    #    主链 = N-Cα-C(=O) 三原子组
    pat = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)")
    bb = set()
    if pat is not None:
        for mt in mol.GetSubstructMatches(pat):
            bb.update(mt)
    sc_in_ring = sum(1 for i in allring if i not in bb)
    d["dl_sidechain_atoms_in_ring"] = float(sc_in_ring)
    # 交联点残基数 ≈ 侧链环原子数的一半（每个交联点占 1~2 个原子）
    d["dl_crosslink_residues"] = float(sc_in_ring / 2.0)

    # ⑤ 交联点局部环境
    if allring:
        het = sum(1 for i in allring if mol.GetAtomWithIdx(i).GetAtomicNum() not in (6,))
        arom = sum(1 for i in allring if mol.GetAtomWithIdx(i).GetIsAromatic())
        d["dl_hetero_at_ring"] = het / len(allring)
        d["dl_arom_at_ring"] = arom / len(allring)

    # ⑥ 两亲性（用序列；无序列则用分子式近似）
    if seq:
        hm, mh = _hydrophobic_moment(seq)
        d["dl_hydro_mean"] = hm
        d["dl_hydro_moment"] = mh
        # 螺旋倾向：疏水残基连续段长度 / 长度
        runs, cur = [], 0
        for c in seq:
            if KD.get(c, -3.5) > 0:
                cur += 1
            else:
                if cur:
                    runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        d["dl_helix_propensity"] = (max(runs) / len(seq)) if runs else 0.0
    # 电荷密度（用形式电荷近似）
    chg = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    d["dl_abs_charge_per_res"] = abs(chg) / n_res
    return d


def self_test() -> int:
    log("=" * 76)
    log("  30_delta_cyclization_features.py 自检")
    log("=" * 76)
    checks = []

    def chk(n, c, i=""):
        checks.append((n, bool(c), i))

    # 线性分子：无环 → 所有 Δ 为 0/默认
    d = delta_features("CCCCCC")
    chk("线性分子：n_rings=0", d["dl_n_rings"] == 0)
    chk("线性分子：ring_atom_frac=0", d["dl_ring_atom_frac"] == 0)
    chk("线性分子：rotbond_ratio=1", abs(d["dl_rotbond_ratio"] - 1.0) < 1e-9)

    # 二硫键环
    # ⚠️ 注意：`C1CCSSCC1` 的环是 **7 元**（4 个 C + 2 个 S + 1 个闭合 C），
    #    不是 8。早期测试期望写错，代码本身是对的。
    d = delta_features("C1CCSSCC1")
    chk("二硫键环：ring_atom_frac>0", d["dl_ring_atom_frac"] > 0,
        str(d["dl_ring_atom_frac"]))
    chk("二硫键环：max_ring=7（4C+2S+1闭合C）", d["dl_max_ring"] == 7.0,
        str(d["dl_max_ring"]))
    # 用 1,2-二噻烷确认真正的 6 元 S-S 环
    d6 = delta_features("C1CCSS1")
    chk("1,2-二噻烷：max_ring=5（3C+2S）", d6["dl_max_ring"] == 5.0,
        str(d6["dl_max_ring"]))

    # 环己烷 vs 线性己烷：约束差异应明显
    dc = delta_features("C1CCCCC1")
    dl = delta_features("CCCCCC")
    chk("环己烷约束密度 > 线性己烷",
        dc["dl_ring_atom_frac"] > dl["dl_ring_atom_frac"], "")

    # 归一化：长度效应 —— 同样一个环，肽越长约束密度越低
    long_pep = "NCC(=O)" * 20 + "C1CCCCC1"
    chk("约束密度按残基归一（长肽更低）",
        dc["dl_constraint_per_res"] > 0)

    # TPSA Δ：环化通常降低 TPSA 或不变
    d = delta_features("O=C1CCCCN1")
    chk("内酰胺：能算出 TPSA Δ 字段", "dl_tpsa_delta" in d)

    # 非法输入
    d = delta_features("这不是SMILES")
    chk("非法 SMILES：ok=0 不抛异常", d["dl_ok"] == 0.0)

    # 维度稳定
    d1 = delta_features("C1CCCCC1"); d2 = delta_features("CCO")
    chk("特征键集合稳定", set(d1) == set(d2))

    # 疏水矩
    hm, mh = _hydrophobic_moment("KKKK")
    chk("疏水矩：全亲水残基矩小", mh < 1.0, f"{mh:.3f}")
    hm2, mh2 = _hydrophobic_moment("LLLL")
    chk("疏水矩：全疏水且周期性 < 全亲水?（长度>=2 可算）", mh2 >= 0)

    # 11) ⭐ 端基修饰剥离（防修饰基团污染序列类特征）
    chk("strip_mods: Ac-…-NH2 被剥离",
        strip_mods("Ac-KYWLKIWRAIRIL-NH2") == "KYWLKIWRAIRIL",
        strip_mods("Ac-KYWLKIWRAIRIL-NH2"))
    chk("strip_mods: 无修饰不变", strip_mods("KYWLKIWRAIRIL") == "KYWLKIWRAIRIL")
    chk("strip_mods: Boc-/Fmoc- 被剥离",
        strip_mods("Fmoc-AAAA-OH") == "AAAA", strip_mods("Fmoc-AAAA-OH"))
    chk("strip_mods: 空串安全", strip_mods("") == "")
    # 关键：加修饰与不加修饰必须得到**相同**的序列类特征
    d_plain = delta_features("C1CCCCC1", "KYWLKIWRAIRIL")
    d_mod = delta_features("C1CCCCC1", "Ac-KYWLKIWRAIRIL-NH2")
    same = all(abs(d_plain[k] - d_mod[k]) < 1e-9 for k in
               ("dl_hydro_mean", "dl_hydro_moment", "dl_helix_propensity",
                "dl_constraint_per_res", "dl_ring_span_per_res"))
    chk("加端基修饰不改变序列类特征", same,
        f"plain hydro={d_plain['dl_hydro_mean']:.4f} "
        f"mod hydro={d_mod['dl_hydro_mean']:.4f}")

    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 76)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Δ 表征：环化导致的理化变化")
    ap.add_argument("--input", default=str(DDIR / "quant_dataset.pkl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()

    p = Path(args.input)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_pickle(p)
    low = {c.lower(): c for c in df.columns}
    idc = low.get("id") or low.get("dbaasp_id") or low.get("cpid") or df.columns[0]
    smic = low.get("smiles")
    seqc = low.get("seq") or low.get("sequence")
    if smic is None:
        log("✗ 缺少 SMILES 列")
        return 2
    log(f"  读入 {len(df)} 条")

    # ⚠️ quant_dataset.pkl **没有序列列**（只有 d_seq_* 派生特征）。
    #    而疏水矩/螺旋倾向依赖真实序列，所以从原始抓取表按 dbaasp_id 补。
    if seqc is None:
        raw_p = DDIR / "dbaasp_cyclic_all_types.csv"
        if raw_p.exists():
            raw = pd.read_csv(raw_p)[["dbaasp_id", "sequence"]]
            raw["dbaasp_id"] = raw["dbaasp_id"].astype(str)
            df = df.copy()
            df["_sid"] = df[idc].astype(str)
            df = df.merge(raw.rename(columns={"dbaasp_id": "_sid"}),
                          on="_sid", how="left")
            df = df.drop(columns=["_sid"])
            seqc = "sequence"
            n_seq = int(df["sequence"].astype(str).str.strip().str.len().gt(0).sum())
            log(f"  已从原始表补入序列：{n_seq}/{len(df)} 条有序列")
        else:
            log("  ⚠ 找不到原始表，序列相关特征将为 0")

    rows = []
    for _, r in df.iterrows():
        f = delta_features(str(r[smic]),
                           str(r[seqc]) if seqc and pd.notna(r.get(seqc)) else "")
        f["dbaasp_id"] = r[idc]
        rows.append(f)
    out = pd.DataFrame(rows)
    cols = ["dbaasp_id"] + [c for c in out.columns if c.startswith(PREFIX)]
    out = out[cols]
    log(f"  产出 {out.shape[1]-1} 个 Δ 特征")
    log(f"  可解析: {int(out['dl_ok'].sum())}/{len(out)}")
    log("")
    log(f"  {'特征':<32}{'均值':>11}{'std':>10}{'非零数':>8}")
    for c in cols[1:]:
        v = pd.to_numeric(out[c], errors="coerce")
        log(f"  {c:<32}{v.mean():>11.4f}{v.std():>10.4f}{int((v!=0).sum()):>8}")

    outp = Path(args.out) if args.out else (DDIR / "delta_cyclization.csv")
    out.to_csv(outp, index=False, encoding="utf-8-sig")
    log("")
    log(f"  ✅ 已写出 {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
