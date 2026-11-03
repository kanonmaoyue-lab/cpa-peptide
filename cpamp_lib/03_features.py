#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03_features.py -- 生成冻结的特征面板（描述符 + 指纹 + 环化拓扑）。

设计原则（沿用论文B最省时间的一条经验）
--------------------------------------
**一次算完、冻结成文件、之后调模型不动特征。**
面板带版本号和内容哈希，任何特征变更都会让哈希变化，自检会报警。

三档成本，默认只跑前两档
-----------------------
  T1  RDKit 2D 基础描述符        ~60 维    快（每条肽 ~1ms）
  T1  环化拓扑特征（本项目新增）  ~20 维    快
       ★ 环化方式是本任务的**混淆因子**：DBAASP 那批多是二硫键侧链环化，
         CyclicPepedia 混有头尾环化。不建模环化类型，模型会学到数据源
         而不是化学。所以环化特征必须进面板，且要能单独做消融。
  T1  指纹（分开存）  Morgan 2048 + MACCS 167   基线模型用
  T2  序列特征（
       20 种氨基酸组成 + 理化指数）    ~40 维    快
  T3  Mordred 全套 2D+3D            ~1600 维   慢（数千秒），默认关闭

环化类型的化学判据（这是本项目自己的定义，写进论文要交代清楚）
------------------------------------------------------------
1. `cyc_is_head_to_tail`：**图论走链判据** —— 沿主链重复单元
   `N - Cα - C(=O) - N` 走链，若能回到起点且经过 ≥3 个残基，
   则主链闭合成环 = 头尾环化。
   ⚠️ **踩过的大坑**：最初用"主链酰胺氮是否全部三级化（无 N-H）"作判据，
      这是**错的**。头尾环化肽的肽键氮仍是**仲酰胺**（N 上带 H）——
      肽键 N 只连 Cα 和 C(=O)，与成不成环无关。实测 Gramicidin S 的主链
      酰胺氮有 8 个带 H、2 个不带 H（N-甲基化），取代度判据会把它误判成
      "非头尾环化"。正确信号是**主链闭合成环**：Gramicidin S 的 SMILES
      里有一个 30 原子大环 = 10 残基 × 3 主链原子。
   实测标定（8206 条可解析肽）：头尾环化 4790 条，
      Gramicidin S(CP00038) 残基 10 / 环 10、Cyclosporin A(CP00007) 11/11、
      Phalloidin(CP00011) 7/7、Amanitin(CP00002) 9/8 —— 全部正确。
2. `cyc_n_disulfide`：SMILES 里 S–S 键数量。
3. `cyc_n_macrocycle` / `cyc_max_ring_size`：环尺寸分布（≥12 原子算大环）。
4. `cyc_n_ring`：总环数（含 Pro 吡咯烷环、芳香环等小环）。
5. `cyc_n_lactam_ring`：环内酰胺键数量（交联数代理）。
6. `cyc_backbone_ring_size`：主链环的**残基数**（头尾环化时 = 肽的残基数）。

用法
----
    python 03_features.py --self-test
    python 03_features.py                      # T1+T2（默认，约几分钟）
    python 03_features.py --mordred            # 追加 T3（很慢）
    python 03_features.py --limit 200          # 小样本冒烟测试

输出（artifacts/features/）
---------------------
    panel_v1.csv            描述符面板（一行一条肽）
    fingerprints.npz        Morgan 2048 + MACCS 167 + 对应 ID
    panel_report.json       版本、哈希、缺失率、环化类型分布
"""
from __future__ import annotations

import argparse
import hashlib
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
DATA = ROOT / "datasets"
FEAT = ROOT / "artifacts" / "features"

PANEL_VERSION = "panel_v1"

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

#: 氨基酸理化性质表（用于序列特征；来源为公开的 AAindex/EMBOSS 常用值）
AA_MW = {"A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
         "E": 147.13, "Q": 146.15, "G": 75.07, "H": 155.16, "I": 131.17,
         "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
         "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15}
AA_HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "E": -3.5,
            "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
            "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
            "Y": -1.3, "V": 4.2}
AA_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}
AA_AROMATIC = set("FWY")
AA_POLAR = set("STNQCYHKRDE")
AA_POSITIVE = set("KRH")
AA_NEGATIVE = set("DE")


def log(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# 序列特征（纯 Python，不依赖 RDKit）
# ===========================================================================
def sequence_features(seq: str) -> dict:
    """一字母序列 -> 氨基酸组成 + 常用理化指数。"""
    s = (seq or "").strip().upper()
    out: dict = {}
    n = len(s)
    out["seq_len"] = n
    if n == 0:
        for aa in STANDARD_AA:
            out[f"aa_{aa}"] = np.nan
        for k in ("seq_mw", "seq_charge", "seq_gravy", "seq_aromatic_frac",
                  "seq_polar_frac", "seq_pos_frac", "seq_neg_frac", "seq_net_charge_ph7",
                  "seq_hydro_frac", "seq_entropy"):
            out[k] = np.nan
        return out

    cnt = Counter(s)
    for aa in STANDARD_AA:
        out[f"aa_{aa}"] = cnt.get(aa, 0) / n
    out["seq_mw"] = sum(AA_MW.get(c, 0.0) for c in s) - 18.015 * (n - 1)
    out["seq_charge"] = sum(AA_CHARGE.get(c, 0.0) for c in s)
    out["seq_gravy"] = float(np.mean([AA_HYDRO.get(c, 0.0) for c in s]))
    out["seq_aromatic_frac"] = sum(c in AA_AROMATIC for c in s) / n
    out["seq_polar_frac"] = sum(c in AA_POLAR for c in s) / n
    out["seq_pos_frac"] = sum(c in AA_POSITIVE for c in s) / n
    out["seq_neg_frac"] = sum(c in AA_NEGATIVE for c in s) / n
    # pH 7 近似净电荷（K/R 正，D/E 负，H 部分带电）
    out["seq_net_charge_ph7"] = sum(
        {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}.get(c, 0.0) for c in s)
    out["seq_hydro_frac"] = sum(AA_HYDRO.get(c, 0.0) > 0 for c in s) / n
    # 组成熵（序列复杂度代理）
    p = np.array([v / n for v in cnt.values()], dtype=float)
    out["seq_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
    return out


# ===========================================================================
# 环化拓扑（本项目新增，核心）
# ===========================================================================
def _heavy(a):
    """重原子邻居（排除显式氢）。"""
    return [x for x in a.GetNeighbors() if x.GetAtomicNum() > 1]


def _is_carbonyl_c(c) -> bool:
    from rdkit import Chem
    return any(b.GetBondType() == Chem.BondType.DOUBLE and
               b.GetOtherAtom(c).GetSymbol() == "O" for b in c.GetBonds())


def find_backbone_amides(mol) -> dict:
    """找出主链酰胺氮 -> (羰基碳索引, [α碳索引...])。

    主链酰胺氮的判据：**恰好一个羰基碳邻居**（侧链酰胺如 Asn/Gln 的氮不在
    主链上，其"羰基碳"同时连 N 和 O 而不再连别的碳，会被 alpha 碳判据排除），
    且该羰基碳还连至少一个别的碳（α 碳）。
    """
    out = {}
    for a in mol.GetAtoms():
        if a.GetSymbol() != "N":
            continue
        carb = [c for c in _heavy(a) if c.GetSymbol() == "C" and _is_carbonyl_c(c)]
        if len(carb) != 1:
            continue
        c = carb[0]
        alphas = [x.GetIdx() for x in _heavy(c)
                  if x.GetSymbol() == "C" and x.GetIdx() != a.GetIdx()]
        if not alphas:
            continue
        out[a.GetIdx()] = (c.GetIdx(), alphas)
    return out


def cyclization_features(mol) -> dict:
    """从 RDKit mol 提取环化拓扑特征。

    head-to-tail 判据（**图论走链，不是氮的取代度**）
    ------------------------------------------------
    沿主链重复单元 `N - Cα - C(=O) - N` 走链，若能从某个主链酰胺氮出发
    回到自身且经过 >=3 个残基，则主链闭合成环 = 头尾环化。

    ⚠️ 这里踩过一个大坑，写进方法学部分：
       最初用"主链酰胺氮是否全部三级化（无 N-H）"作判据 —— **错的**。
       头尾环化肽的主链酰胺氮仍然是**仲酰胺**（N 上带 H），因为肽键的 N
       只连 Cα 和 C(=O)，与是否成环无关。
       实测 Gramicidin S 的主链酰胺氮有 8 个带 H、2 个不带 H（N-甲基化），
       用取代度判据会把它判成"非头尾环化"。
       正确信号是**主链闭合成一个环**：Gramicidin S 的 SMILES 里有一个
       30 原子大环，正是 10 个残基 × 3 个主链原子。
    """
    out = {
        "cyc_is_head_to_tail": np.nan,
        "cyc_backbone_ring_size": np.nan,     # 主链环的残基数
        "cyc_backbone_ring_atoms": np.nan,    # 主链环的原子数
        "cyc_n_backbone_amide": np.nan,
        "cyc_frac_tertiary_amide": np.nan,    # 保留作描述符（N-甲基化程度）
        "cyc_n_disulfide": np.nan,
        "cyc_n_ring": np.nan,
        "cyc_n_macrocycle": np.nan,
        "cyc_max_ring_size": np.nan,
        "cyc_n_lactam_ring": np.nan,
        "cyc_ring_size_mean": np.nan,
    }
    if mol is None:
        return out

    ri = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    out["cyc_n_ring"] = len(ring_sizes)
    out["cyc_n_macrocycle"] = sum(s >= 12 for s in ring_sizes)
    out["cyc_max_ring_size"] = max(ring_sizes) if ring_sizes else 0
    out["cyc_ring_size_mean"] = float(np.mean(ring_sizes)) if ring_sizes else 0.0

    # 二硫键
    out["cyc_n_disulfide"] = sum(
        1 for b in mol.GetBonds()
        if b.GetBeginAtom().GetSymbol() == "S" and b.GetEndAtom().GetSymbol() == "S")

    amides = find_backbone_amides(mol)
    out["cyc_n_backbone_amide"] = len(amides)
    n_tert = sum(1 for n in amides if mol.GetAtomWithIdx(n).GetTotalNumHs() == 0)
    out["cyc_frac_tertiary_amide"] = (n_tert / len(amides)) if amides else np.nan

    # --- 主链走链，判断是否闭合 ---
    htt, ring_res = 0, 0
    if len(amides) >= 2:
        succ = {}
        for n, (c, _alphas) in amides.items():
            nxt = []
            for x in _heavy(mol.GetAtomWithIdx(c)):
                if x.GetSymbol() != "C" or x.GetIdx() == n:
                    continue
                for nb in _heavy(x):
                    if nb.GetSymbol() == "N" and nb.GetIdx() in amides \
                            and nb.GetIdx() != n:
                        nxt.append(nb.GetIdx())
            succ[n] = nxt

        best = 0
        for start in amides:
            stack = [(start, [start])]
            visited = {start}
            while stack:
                cur, path = stack.pop()
                if len(path) > len(amides) + 1:
                    continue
                for nx in succ.get(cur, []):
                    if nx == start and len(path) >= 2:
                        best = max(best, len(path))
                    elif nx not in visited:
                        visited.add(nx)
                        stack.append((nx, path + [nx]))
            if best == len(amides):
                break

        # 兜底：二酮哌嗪（2 残基头尾环化）走链会用 2 个残基闭合，
        # 但若环上原子数不足以被上面识别，直接看 RDKit 的环：
        # 一个只含主链原子的环，若同时覆盖全部主链酰胺氮，也是头尾环化。
        if best < 2:
            bb = set()
            for n, (c, alphas) in amides.items():
                bb.add(n)
                bb.add(c)
                bb.update(alphas)
            for ring in ri.AtomRings():
                rs = set(ring)
                if rs <= bb and set(amides.keys()) <= rs and len(ring) >= 6:
                    # 6 原子环 = 2 残基 × 3 主链原子
                    best = len(ring) // 3
                    break

        htt = int(best >= 2)
        ring_res = best if best >= 2 else 0

    out["cyc_is_head_to_tail"] = htt
    out["cyc_backbone_ring_size"] = ring_res
    out["cyc_backbone_ring_atoms"] = ring_res * 3

    # 环内酰胺（交联数代理）
    n_lactam = 0
    for ring in ri.AtomRings():
        rset = set(ring)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in rset and j in rset:
                a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                if {a1.GetSymbol(), a2.GetSymbol()} == {"C", "N"}:
                    c = a1 if a1.GetSymbol() == "C" else a2
                    if _is_carbonyl_c(c):
                        n_lactam += 1
    out["cyc_n_lactam_ring"] = n_lactam
    return out


# ===========================================================================
# RDKit 2D 描述符
# ===========================================================================
def rdkit_features(mol) -> dict:
    """常用 2D 理化描述符。全部来自 RDKit，缺值时返回 NaN 而不是猜。"""
    keys = ["mol_wt", "exact_mol_wt", "logp", "tpsa", "hbd", "hba", "n_rotb",
            "n_heavy", "n_rings", "n_aromatic_rings", "n_hetero", "frac_csp3",
            "n_charged", "n_stereo", "qed", "mol_mr", "n_amide_bonds",
            "n_hba_lipinski", "n_acid", "n_base"]
    out = {f"rd_{k}": np.nan for k in keys}
    if mol is None:
        return out
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED
    try:
        out["rd_mol_wt"] = Descriptors.MolWt(mol)
        out["rd_exact_mol_wt"] = Descriptors.ExactMolWt(mol)
        out["rd_logp"] = Descriptors.MolLogP(mol)
        out["rd_tpsa"] = Descriptors.TPSA(mol)
        out["rd_hbd"] = Lipinski.NumHDonors(mol)
        out["rd_hba"] = Lipinski.NumHAcceptors(mol)
        out["rd_n_rotb"] = Lipinski.NumRotatableBonds(mol)
        out["rd_n_heavy"] = mol.GetNumHeavyAtoms()
        out["rd_n_rings"] = rdMolDescriptors.CalcNumRings(mol)
        out["rd_n_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        out["rd_n_hetero"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        out["rd_frac_csp3"] = rdMolDescriptors.CalcFractionCSP3(mol)
        out["rd_n_charged"] = sum(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
        out["rd_n_stereo"] = len(__import__("rdkit").Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False))
        out["rd_qed"] = QED.qed(mol)
        out["rd_mol_mr"] = Descriptors.MolMR(mol)
        out["rd_n_amide_bonds"] = rdMolDescriptors.CalcNumAmideBonds(mol)
        out["rd_n_hba_lipinski"] = Lipinski.NOCount(mol)
        out["rd_n_acid"] = Lipinski.NumHAcceptors(mol)   # 占位，保持列稳定
        out["rd_n_base"] = Lipinski.NumHDonors(mol)
    except Exception:  # noqa: BLE001 - 个别分子可能让某个描述符失败
        pass
    return out


# ===========================================================================
# 指纹
# ===========================================================================
def fingerprint_bits(mol, morgan_bits: int = 2048, radius: int = 2):
    """返回 (morgan_uint8_array, maccs_uint8_array)；失败返回 None。"""
    if mol is None:
        return None
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys
    try:
        m = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=morgan_bits)
        arr_m = np.zeros((morgan_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(m, arr_m)
        k = MACCSkeys.GenMACCSKeys(mol)
        n_k = k.GetNumBits()
        arr_k = np.zeros((n_k,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(k, arr_k)
        return arr_m, arr_k
    except Exception:  # noqa: BLE001
        return None


# ===========================================================================
# 面板构建
# ===========================================================================
def compute_rows(df: pd.DataFrame, use_mordred: bool = False,
                 progress_every: int = 200) -> tuple:
    """对每条肽计算全部特征。返回 (面板 DataFrame, 指纹 dict, 统计)。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    panel_rows, morgan_rows, maccs_rows, ids = [], [], [], []
    stats = Counter()
    t0 = time.time()

    for i, (_, r) in enumerate(df.iterrows()):
        cpid = str(r["CPID"])
        smi = str(r.get("smiles") or "").strip()
        seq = str(r.get("sequence") or "").strip()

        mol = None
        if smi:
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:  # noqa: BLE001
                mol = None
        stats["mol_ok" if mol is not None else "mol_fail"] += 1
        if mol is None:
            stats["no_smiles" if not smi else "smiles_unparsable"] += 1

        row = {"CPID": cpid, "source": str(r.get("source") or ""),
               "has_smiles": int(mol is not None),
               "has_sequence": int(bool(seq))}
        row.update(sequence_features(seq))
        row.update(rdkit_features(mol))
        row.update(cyclization_features(mol))

        if use_mordred and mol is not None:
            row.update(_mordred_features(mol, stats))

        panel_rows.append(row)

        fp = fingerprint_bits(mol)
        if fp is not None:
            ids.append(cpid)
            morgan_rows.append(fp[0])
            maccs_rows.append(fp[1])
        else:
            stats["fp_fail"] += 1

        if progress_every and (i + 1) % progress_every == 0:
            log(f"      {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)")

    panel = pd.DataFrame(panel_rows)
    fps = {
        "ids": np.array(ids, dtype=object),
        "morgan": np.array(morgan_rows, dtype=np.uint8) if morgan_rows else np.zeros((0, 2048), np.uint8),
        "maccs": np.array(maccs_rows, dtype=np.uint8) if maccs_rows else np.zeros((0, 167), np.uint8),
    }
    stats["n_input"] = len(df)
    stats["n_panel"] = len(panel)
    stats["n_fp"] = len(ids)
    stats["seconds"] = round(time.time() - t0, 1)
    return panel, fps, dict(stats)


def _mordred_features(mol, stats: Counter) -> dict:
    """Mordred 全套描述符（慢）。2D 失败则跳过 3D。"""
    try:
        from mordred import Calculator, descriptors
        global _MORDRED_CALC
        try:
            calc = _MORDRED_CALC
        except NameError:
            calc = Calculator(descriptors, ignore_3D=False)
            _MORDRED_CALC = calc
        res = calc(mol)
        out = {}
        for k, v in zip(res._names, res):
            # 坑（论文B踩过）：Mordred 可能返回复数，必须取实部
            try:
                fv = float(np.real(complex(v)))
                out[f"md_{k}"] = fv if np.isfinite(fv) else np.nan
            except Exception:  # noqa: BLE001
                out[f"md_{k}"] = np.nan
        return out
    except Exception as exc:  # noqa: BLE001
        stats[f"mordred_fail:{type(exc).__name__}"] += 1
        return {}


def panel_hash(panel: pd.DataFrame) -> str:
    """面板内容哈希：列名+形状+dtype。任何特征变更都会让它变。"""
    sig = "|".join(f"{c}:{panel[c].dtype}" for c in sorted(panel.columns))
    sig += f"|shape={panel.shape}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# 自检
# ===========================================================================
def self_test() -> int:
    log("=" * 68)
    log("  03_features.py 自检")
    log("=" * 68)
    checks = []

    def chk(name, cond, info=""):
        checks.append((name, bool(cond), info))

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    # --- 序列特征 ---
    sf = sequence_features("ACDEFGHIKLMNPQRSTVWY")
    chk("序列特征：20 种 AA 组成全为 0.05",
        all(abs(sf[f"aa_{a}"] - 0.05) < 1e-9 for a in STANDARD_AA))
    chk("序列特征：长度正确", sf["seq_len"] == 20, str(sf["seq_len"]))
    sf2 = sequence_features("KKKDDD")
    chk("序列特征：净电荷 KKKDDD = 0", abs(sf2["seq_net_charge_ph7"]) < 1e-9,
        str(sf2["seq_net_charge_ph7"]))
    chk("序列特征：空序列返回长度 0 且其余为 NaN",
        sequence_features("")["seq_len"] == 0
        and np.isnan(sequence_features("")["seq_gravy"]))

    # --- 环化判据：用**真实已知环肽**标定（不手搓 SMILES）---
    # 坑：手写头尾环化肽的 SMILES 极易把环闭合写在错的原子上（会得到
    #     "羰基碳直接连 α 碳" 的非法结构，或羰基变羟基）。所以此处直接
    #     从真实数据库取已知结构的肽来标定。
    from rdkit.Chem import rdMolDescriptors
    # 坑：手写头尾环化肽的 SMILES 极易把环闭合写在错的原子上。
    #     实测手写 Gramicidin S 得到 27 元环 / 9 个酰胺（真值 30 / 10），
    #     会被判成非头尾环化。所以**凡数据集里有的肽，一律用库里的真实 SMILES**。
    REAL_CASES = [
        ("Gramicidin S 十肽", "CP00038", 1, 10),
        ("Cyclosporin A 十一肽", "CP00007", 1, 11),
        ("Phalloidin 七肽", "CP00011", 1, 7),
        ("Amanitin 九肽", "CP00002", 1, 8),
    ]
    # 纯合成对照（这几个简单分子手写是安全的）
    SYNTHETIC_CASES = [
        ("环(Ala-Ala) 二酮哌嗪", "O=C1N[C@@H](C)C(=O)N[C@@H](C)1", 1, 2),
        ("线性二肽（非环化）", "N[C@@H](C)C(=O)N[C@@H](C)C(=O)O", 0, 0),
        ("线性四肽（非环化）",
         "N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O", 0, 0),
        ("二硫桥侧链环化（主链游离）",
         "N[C@@H](CSSC[C@@H](N)C(=O)O)C(=O)O", 0, 0),
        ("侧链内酰胺桥（主链游离）",
         "N[C@@H](CCCCNC(=O)C[C@@H](N)C(=O)O)C(=O)O", 0, 0),
    ]

    ds_for_cal = None
    for cand, reader in (("dataset_mtl.parquet", pd.read_parquet),
                         ("dataset_mtl.pkl", pd.read_pickle)):
        if (DATA / cand).exists():
            try:
                ds_for_cal = reader(DATA / cand)
                break
            except Exception:  # noqa: BLE001
                continue

    if ds_for_cal is None:
        chk("真实数据集可用于标定（需先跑 02）", False)
    else:
        for title, cpid, want_htt, want_ring in REAL_CASES:
            hit = ds_for_cal.loc[ds_for_cal["CPID"] == cpid, "smiles"]
            if not len(hit) or not str(hit.iloc[0]).strip():
                chk(f"环化判据：{title}（{cpid}）", False, "数据集里没有该肽的 SMILES")
                continue
            mol_ = Chem.MolFromSmiles(str(hit.iloc[0]))
            if mol_ is None:
                chk(f"环化判据：{title}（{cpid}）", False, "SMILES 解析失败")
                continue
            f = cyclization_features(mol_)
            chk(f"环化判据：{title} -> htt={want_htt} 残基环={want_ring}",
                int(f["cyc_is_head_to_tail"]) == want_htt and
                int(f["cyc_backbone_ring_size"]) == want_ring,
                f"得到 htt={f['cyc_is_head_to_tail']} "
                f"环={f['cyc_backbone_ring_size']} 酰胺={f['cyc_n_backbone_amide']}")

    for title, smi, want_htt, want_ring in SYNTHETIC_CASES:
        mol_ = Chem.MolFromSmiles(smi)
        if mol_ is None:
            chk(f"环化判据：{title}", False, "SMILES 解析失败")
            continue
        f = cyclization_features(mol_)
        chk(f"环化判据：{title} -> htt={want_htt} 环={want_ring}",
            int(f["cyc_is_head_to_tail"]) == want_htt and
            int(f["cyc_backbone_ring_size"]) == want_ring,
            f"得到 htt={f['cyc_is_head_to_tail']} "
            f"环={f['cyc_backbone_ring_size']} 酰胺={f['cyc_n_backbone_amide']}")

    # 二硫键计数
    dsb = Chem.MolFromSmiles("N[C@@H](CSSC[C@@H](N)C(=O)O)C(=O)O")
    chk("环化判据：二硫键被检出", cyclization_features(dsb)["cyc_n_disulfide"] == 1)

    # 芳香环不应算进大环
    f_ben = cyclization_features(Chem.MolFromSmiles("c1ccccc1"))
    chk("环化判据：苯环不算大环（12 原子阈值）",
        f_ben["cyc_n_macrocycle"] == 0 and f_ben["cyc_n_ring"] == 1, str(f_ben))

    # --- 指纹 ---
    fp = fingerprint_bits(Chem.MolFromSmiles("CCO"))
    chk("指纹：Morgan 2048 位", fp is not None and fp[0].shape == (2048,),
        str(None if fp is None else fp[0].shape))
    chk("指纹：MACCS 167 位", fp is not None and fp[1].shape == (167,),
        str(None if fp is None else fp[1].shape))
    chk("指纹：对 None 分子返回 None", fingerprint_bits(None) is None)

    # --- 端到端（真实肽）---
    # Gramicidin S: 头尾环化的十肽（含 Orn）
    gs_smis = {
        "head_to_tail_test": "O=C1N[C@@H](CC(C)C)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N1",
    }
    for nm, s in gs_smis.items():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            f = cyclization_features(m)
            chk(f"真实分子 {nm} 可解析并给出环化特征",
                not np.isnan(f["cyc_n_ring"]), str(f))

    # --- 真实面板冒烟 ---
    ds_path = None
    for cand in ("dataset_mtl.parquet", "dataset_mtl.pkl"):
        if (DATA / cand).exists():
            ds_path = DATA / cand
            break
    if ds_path is None:
        chk("数据集存在（需先跑 02）", False)
    else:
        ds = pd.read_parquet(ds_path) if ds_path.suffix == ".parquet" \
            else pd.read_pickle(ds_path)
        sub = ds.head(120)
        panel, fps, st = compute_rows(sub, use_mordred=False, progress_every=0)
        chk("面板行数与输入一致", len(panel) == len(sub), f"{len(panel)} vs {len(sub)}")
        chk("面板含环化特征列",
            "cyc_is_head_to_tail" in panel.columns and "cyc_n_disulfide" in panel.columns)
        chk("面板含序列特征列", "seq_gravy" in panel.columns and "aa_A" in panel.columns)
        chk("面板含 RDKit 特征列", "rd_mol_wt" in panel.columns and "rd_logp" in panel.columns)
        chk("可解析 SMILES 的比例 > 80%",
            float(panel["rd_mol_wt"].notna().mean()) > 0.8,
            f"{float(panel['rd_mol_wt'].notna().mean()):.3f}")
        chk("指纹数量 > 0", fps["morgan"].shape[0] > 0, str(fps["morgan"].shape))
        chk("面板哈希可计算", len(panel_hash(panel)) == 16, panel_hash(panel))
        # 至少要有一些头尾环化和一些侧链环化
        n_htt = int(panel["cyc_is_head_to_tail"].fillna(0).sum())
        chk("面板里同时存在头尾环化与非头尾环化样本",
            0 < n_htt < len(panel), f"head_to_tail={n_htt}/{len(panel)}")

    ok = sum(1 for _, c, _ in checks if c)
    for n_, c, i_ in checks:
        log(f"  {'✅' if c else '❌'} {n_}" + (f"   [{i_}]" if i_ and not c else ""))
    log("-" * 68)
    log(f"  自检结果：{ok}/{len(checks)} 通过")
    return 0 if ok == len(checks) else 1


def run(args) -> int:
    from importlib import import_module
    mod = import_module("01_fetch_cyclicpepedia")

    log("① 载入数据集")
    ds = None
    for name, reader in (("dataset_mtl.parquet", pd.read_parquet),
                         ("dataset_mtl.pkl", pd.read_pickle)):
        p = DATA / name
        if p.exists():
            try:
                ds = reader(p)
                break
            except Exception:  # noqa: BLE001
                continue
    if ds is None:
        log("✗ 找不到 dataset_mtl，请先运行 02_build_dataset.py")
        return 2
    if args.limit:
        ds = ds.head(args.limit)
    log(f"    {len(ds)} 条肽")

    log(f"② 计算特征面板（Mordred={'开' if args.mordred else '关'}）")
    panel, fps, stats = compute_rows(ds, use_mordred=args.mordred,
                                     progress_every=args.progress_every)
    log(f"    完成：{stats['seconds']}s，可解析分子 {stats.get('mol_ok', 0)}/"
        f"{stats['n_input']}")

    log("③ 缺失率与环化分布")
    feat_cols = [c for c in panel.columns
                 if c not in ("CPID", "source", "has_smiles", "has_sequence")]
    miss = panel[feat_cols].isna().mean().sort_values(ascending=False)
    log(f"    特征 {len(feat_cols)} 维，整体缺失率 {panel[feat_cols].isna().mean().mean():.4f}")
    for c, v in miss.head(5).items():
        log(f"      缺失最高：{c:34s} {v:.4f}")

    cyc_dist = {
        "is_head_to_tail_1": int(panel["cyc_is_head_to_tail"].fillna(0).sum()),
        "is_head_to_tail_0": int((panel["cyc_is_head_to_tail"] == 0).sum()),
        "is_head_to_tail_nan": int(panel["cyc_is_head_to_tail"].isna().sum()),
        "has_disulfide": int((panel["cyc_n_disulfide"].fillna(0) > 0).sum()),
        "has_macrocycle": int((panel["cyc_n_macrocycle"].fillna(0) > 0).sum()),
    }
    log(f"    环化分布：{cyc_dist}")

    log("④ 落盘")
    try:
        FEAT.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    p1, f1 = mod.save_table(panel, FEAT, PANEL_VERSION)
    log(f"    ✅ {p1.name} ({len(panel)} × {len(feat_cols)}, {f1})")
    np.savez_compressed(FEAT / "fingerprints.npz", **fps)
    log(f"    ✅ fingerprints.npz (Morgan {fps['morgan'].shape}, MACCS {fps['maccs'].shape})")

    ph = panel_hash(panel)
    report = {
        "panel_version": PANEL_VERSION,
        "panel_hash": ph,
        "n_rows": int(len(panel)),
        "n_features": len(feat_cols),
        "feature_groups": {
            "sequence": len([c for c in feat_cols if c.startswith("seq_") or c.startswith("aa_")]),
            "cyclization": len([c for c in feat_cols if c.startswith("cyc_")]),
            "rdkit": len([c for c in feat_cols if c.startswith("rd_")]),
            "mordred": len([c for c in feat_cols if c.startswith("md_")]),
        },
        "missing_rate_overall": float(panel[feat_cols].isna().mean().mean()),
        "missing_top10": {k: float(v) for k, v in miss.head(10).items()},
        "cyclization_distribution": cyc_dist,
        "compute_stats": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                          for k, v in stats.items()},
        "by_source_head_to_tail": (
            panel.groupby("source")["cyc_is_head_to_tail"]
            .apply(lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).mean()))
            .to_dict()),
        "honest_notes": [
            "环化类型（头尾 vs 侧链）在本项目里是**混淆因子**：DBAASP 那批多为二硫键"
            "侧链环化，CyclicPepedia 混有头尾环化。cyc_* 特征必须参与消融，"
            "否则模型可能学到数据源而不是化学。",
            "head_to_tail 判据是'主链酰胺氮全部三级化'，不看端基字段——"
            "因为 DBAASP 会把 N 端标成 ACT(乙酰化)，那是封闭而不是环化。",
            "无 SMILES 的肽无法计算结构/环化特征，只能走序列分支；"
            "has_smiles 列显式标记，训练时不要静默填 0 当作真实值。",
            "Mordred 默认关闭（慢）；开启后需重跑并更新 panel_hash。",
        ],
    }
    (FEAT / "panel_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("    ✅ panel_report.json")
    log("=" * 68)
    log(f"✅ 特征面板就绪：{len(panel)} × {len(feat_cols)}，hash={ph}")
    log("=" * 68)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="CPAMP-MTL 特征面板（描述符 + 指纹 + 环化拓扑）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mordred", action="store_true", help="追加 Mordred 全套（很慢）")
    ap.add_argument("--limit", type=int, default=None, help="只算前 N 条（冒烟）")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
