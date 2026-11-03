"""cpamp_predict_v2.py -- 与论文一致的推理接口：cfu + ESM 融合 + 区间 + 适用域。

═══ 与旧 cpamp_predict.py 的差别 ═══
    旧版：输入 2 331 + 6 = 2 337，无 cfu、无 ESM、无区间、适用域只看结构相似度。
    本版：输入 2 331 + cfu 2 + 6 = 2 339，与 ESM(960+6) 按 w=0.35 融合，
          输出 90% 区间，适用域由「菌株 / 环化类型 / 结构 / 条件」四轴共同决定。

═══ 输入列（除 SMILES 外均可选）═══
    SMILES        必需
    Seq           可选（缺失时序列特征置缺失）
    Strain        可选（缺失或未知 → 只输出定性）
    cfuGroup      可选，取值 0–4（<1E4 … >1E7）
    Medium        可选，如 'MHB' / 'CAMHB'
    CyclizationType  可选（如 'DSB'）；缺失视为未知

═══ 输出列 ═══
    定量_log10uM / 定量_uM                点预测
    定量_uM_low90 / 定量_uM_high90        90% 区间
    σ_log10                               所用区间宽度
    新颖度_菌株 / 新颖度_环化 / 结构最大相似度 / 条件是否标准
    信赖等级                               高 / 中 / 低 / 仅定性
    说明                                   逐条理由
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_BUNDLE_CANDS = [ROOT / "models" / "final_bundle.pkl",
                 ROOT / "artifacts" / "final_model" / "final_bundle.pkl"]
BUNDLE = next((q for q in _BUNDLE_CANDS if q.exists()),
              _BUNDLE_CANDS[-1])
ESM_CACHE = ROOT / "artifacts" / "esm2" / "esm2_t12_emb.npz"
Z = 1.6448536269514722  # 90% 双侧
CAL = ROOT / "artifacts" / "qual_calibration.json"

sys.path.insert(0, str(ROOT / "cpamp_lib"))


_BUNDLE_CACHE = {}


def _load_bundle(path=None):
    """path 为 None 时用默认模型；否则按路径缓存加载。"""
    key = str(path) if path else "__default__"
    if key not in _BUNDLE_CACHE:
        fp = Path(path) if path else BUNDLE
        with open(fp, "rb") as fh:
            _BUNDLE_CACHE[key] = pickle.load(fh)
    return _BUNDLE_CACHE[key]


def _esm_embed(seqs, bundle):
    """优先用缓存；未命中则现场计算（需要 transformers）。"""
    cache = {}
    if ESM_CACHE.exists():
        z = np.load(ESM_CACHE, allow_pickle=True)
        # 关键：先取出数组再建字典。
        # 若写成 z["emb"][i]，每次迭代都会重新解压整个 npz 成员（3094 次 ≈ 200 秒）
        _emb = z["emb"]
        _seq = [str(s) for s in z["seq"]]
        cache = {s: _emb[i] for i, s in enumerate(_seq)}
        del z
    dim = 960
    out = np.zeros((len(seqs), dim), np.float32)
    miss = []
    for i, s in enumerate(seqs):
        s = (s or "").strip().upper()
        if s in cache:
            out[i] = cache[s]
        else:
            out[i] = np.nan
            miss.append((i, s))
    # 只对**有实际序列**的缺失项计算；全空时直接跳过，避免白白加载模型
    miss = [(i, s) for i, s in miss if s]
    if miss:
        try:
            # 优先用本地 HF 缓存，避免联网
            for env, val in (("HF_HOME", str(ROOT / "artifacts" / "esm2" / "hf_home")),
                             ("HF_HUB_OFFLINE", "1"),
                             ("TRANSFORMERS_OFFLINE", "1")):
                os.environ.setdefault(env, val)
            import torch
            from transformers import AutoModel, AutoTokenizer
            name = bundle["esm_model_name"]
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name).eval()
            for i, s in miss:
                if not s:
                    continue
                with torch.no_grad():
                    b = tok(s, return_tensors="pt", truncation=True,
                            max_length=1024)
                    h = mdl(**b).last_hidden_state[0].numpy()
                out[i, :dim // 2] = h.mean(0)
                out[i, dim // 2:] = h.max(0)
            ok = True
        except Exception as e:  # pragma: no cover
            print(f"  [warn] ESM 不可用（{type(e).__name__}）："
                  f"回退到主模型单独预测（w_esm → 0）", file=sys.stderr)
            ok = False
            for i, s in miss:
                out[i] = np.nan
    return out


def _esm_available(E):
    """逐行标记是否有真实嵌入（零向量视作缺失）。"""
    return np.isfinite(E).all(1) and True


_QUAL_BUNDLE = {}
_QUAL_CAL = {}


def _load_qual(path=None):
    """定性模型（目标条件化统一模型，3 seeds）。"""
    key = str(path) if path else "__default__"
    if key not in _QUAL_BUNDLE:
        fp = Path(path) if path else (ROOT / "artifacts" / "pantarget"
                                      / "pantarget_models.pkl")
        with open(fp, "rb") as fh:
            _QUAL_BUNDLE[key] = pickle.load(fh)
    return _QUAL_BUNDLE[key]


def _load_cal():
    if not _QUAL_CAL:
        if CAL.exists():
            _QUAL_CAL.update(json.loads(CAL.read_text(encoding="utf-8")))
        else:
            _QUAL_CAL["tasks"] = {}
    return _QUAL_CAL


# 模型内部任务名（中文）→ 校准表键（英文）
_TASK_EN = {"总抗菌(H2)": "Any antibacterial", "总抗菌": "Any antibacterial",
            "抗细菌": "Antibacterial", "抗革兰阳性": "Gram-positive",
            "抗革兰阴性": "Gram-negative", "抗真菌": "Antifungal"}


def _cal_lookup(task, p):
    """概率 → 实测阳性率 + Wilson 95% 区间。

    分箱过稀（n<20）时该箱被剔除。此时对已填充箱的
    (mean_p, obs_rate) 做**线性插值**给出点估计，
    Wilson 区间取最近箱（保守）。exact=False 表示该点非实测箱内。
    """
    key = _TASK_EN.get(task, task)
    t = _load_cal().get("tasks", {}).get(key)
    if not t or not t.get("bins"):
        return None
    bins = t["bins"]
    for b in bins:
        if b["lo"] <= p < b["hi"] or (b["hi"] >= 1.0 and p >= b["lo"]):
            return dict(b, exact=True)
    xs = np.array([b["mean_p"] for b in bins], float)
    ys = np.array([b["obs_rate"] for b in bins], float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    est = float(np.interp(p, xs, ys))
    j = int(np.argmin(np.abs(xs - p)))
    b = bins[o[j]]
    return dict(b, obs_rate=round(est, 4), exact=False)


def _qual_predict(X2277, bundle_path=None):
    """X2277: (n, 2277) → (概率 (n,5), 种子SD (n,5), 任务名列表)"""
    obj = _load_qual()
    models = [v[2] if isinstance(v, tuple) else v for v in obj["models"]]
    tasks = list(obj["tasks"])
    n = X2277.shape[0]
    P3 = np.zeros((len(models), n, 5), np.float32)
    Xv = np.nan_to_num(X2277, nan=0.0, posinf=0.0, neginf=0.0)
    for mi, m in enumerate(models):
        for j in range(5):
            tgt = np.zeros((n, 5), np.float32)
            tgt[:, j] = 1
            P3[mi, :, j] = m.predict_proba(np.hstack([Xv, tgt]))[:, 1]
    return P3.mean(0), P3.std(0), tasks


def _cond_standard(cfu, medium, bundle):
    """条件是否落在标准药敏区间；返回 (是否标准, 是否已知)"""
    std_med = set(bundle["std_medium"])
    std_cfu = set(bundle["std_cfu"])
    ok_m = (medium is not None) and str(medium).strip().upper() in std_med
    ok_c = (cfu is not None) and (not pd.isna(cfu)) and int(cfu) in std_cfu
    known = (medium is not None and str(medium).strip() != "") or \
            (cfu is not None and not pd.isna(cfu))
    return bool(ok_m and ok_c), bool(known)


_STRAIN_SUP = {}


def _strain_support():
    if not _STRAIN_SUP:
        fp = ASSETS / "strain_support.json" if "ASSETS" in globals() else \
            (ROOT / "assets" / "strain_support.json")
        try:
            _STRAIN_SUP.update(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            pass
    return _STRAIN_SUP


def _downgrade(grade, steps):
    order = ["高", "中", "低", "仅定性"]
    if grade not in order:
        return grade
    return order[min(len(order) - 1, order.index(grade) + steps)]


def predict(frame, use_esm=True, calibrate=True, verbose=False, bundle=None, with_qual=True, qual_bundle=None):
    bundle = _load_bundle(bundle)
    import cpamp_predict as cp

    d = frame.copy().reset_index(drop=True)
    feats, keep, resolved = cp.build_features(
        d["SMILES"].fillna("").tolist(),
        d["Seq"].fillna("").tolist() if "Seq" in d else [""] * len(d))
    out = d.copy()
    out["状态"] = "SMILES 无法解析"
    if len(keep) == 0:
        return out
    out.loc[keep, "状态"] = "成功"

    X = feats["fixed"]                       # (n, 2331)
    n = len(keep)

    # ── cfu ──
    cfu_col = None
    for c in ("cfuGroup", "cfu_group", "CFU", "cfu"):
        if c in d:
            cfu_col = c
            break
    cfu = np.array([pd.to_numeric(d.loc[i, cfu_col], errors="coerce")
                    if cfu_col else np.nan for i in keep], float)
    cfu_f = np.where(np.isfinite(cfu), cfu, bundle["cfu_fill"])

    # ── 菌株上下文 ──
    strain_col = "Strain" if "Strain" in d else None
    targets = [str(d.loc[i, strain_col]).strip()
               if strain_col and pd.notna(d.loc[i, strain_col]) else ""
               for i in keep]
    codes, meds = bundle["strain_codes"], bundle["strain_median"]
    groups = bundle["strain_group"]
    order = ["gram_pos", "gram_neg", "fungal", "other"]
    C = np.zeros((n, 6), np.float32)
    for i, t in enumerate(targets):
        g = groups.get(t, "other")
        C[i, order.index(g) if g in order else 3] = 1
        C[i, 4] = codes.get(t, len(codes))          # 未知 → 专用类别
        C[i, 5] = meds.get(t, np.nanmedian(list(meds.values())))
    known_strain = np.array([bool(t) and t in codes for t in targets])

    XM = np.hstack([X, cfu_f[:, None],
                    (~np.isfinite(cfu)).astype(np.float32)[:, None], C])
    pm = np.mean([m.predict(XM) for m in bundle["main_models"]], axis=0)

    # ── ESM ──
    esm_used = 0
    if use_esm:
        E = _esm_embed([resolved[i] if i < len(resolved) else ""
                        for i in range(n)], bundle)
        has_e = np.isfinite(E).all(1)
        esm_used = int(has_e.sum())
        w = bundle["w_esm"]
        pred = pm.copy()
        if esm_used:
            XE = np.hstack([np.nan_to_num(E, nan=0.0), C])
            pe = np.mean([m.predict(XE) for m in bundle["esm_models"]], axis=0)
            pred[has_e] = (1 - w) * pm[has_e] + w * pe[has_e]
    else:
        pred = pm

    # ── 校准（§4.5.5 分区收缩）──
    type_col = None
    for c in ("CyclizationType", "cyclization_type", "Bond", "bond"):
        if c in d:
            type_col = c
            break
    # ⚠️ 环化类型**不从 SMILES 推断**做校准：
    #    训练标签用的是 DBAASP 的 18 类（DSB/AMD/EST/...），
    #    而 SMILES 规则只能给出 5 类粗略划分（HT/MIX/SC/DSB/OTHER），
    #    两者不兼容，误用会让所有输入都被判为「新类型」。
    #    未提供类型时一律按未知处理（k = 0.51，σ 取 locto 档）。
    if type_col is not None:
        types = [str(d.loc[i, type_col]).strip() for i in keep]
    else:
        types = ["" for _ in keep]
    known_type = np.array([t in set(bundle["cyc_types"]) for t in types])
    if calibrate:
        k = np.where(known_type, 1.0, 0.51)
        pred_cal = pred * k
    else:
        pred_cal = pred

    # ── 结构新颖度 ──
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    ref = bundle["fp_ref"]
    Q = np.zeros((n, ref.shape[1]), np.float32)
    for i in range(n):
        m = Chem.MolFromSmiles(str(d.loc[keep[i], "SMILES"]))
        if m is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, ref.shape[1])
            for b in fp.GetOnBits():
                Q[i, b] = 1.0
    inter = Q @ ref.T
    szq, szr = Q.sum(1), ref.sum(1)
    T = inter / np.maximum(szq[:, None] + szr[None, :] - inter, 1e-9)
    sim = T.max(1) if T.size else np.zeros(n)

    # ── 条件 ──
    med_col = None
    for c in ("Medium", "medium"):
        if c in d:
            med_col = c
            break
    std_flag = np.zeros(n, bool)
    cond_known = np.zeros(n, bool)
    for i in range(n):
        mv = str(d.loc[keep[i], med_col]).strip() if med_col else ""
        cv = cfu[i] if np.isfinite(cfu[i]) else None
        std_flag[i], cond_known[i] = _cond_standard(cv, mv or None, bundle)

    # ── 区间 ──
    sig = np.zeros(n)
    for i in range(n):
        if not known_type[i]:
            key = ("locto", bool(std_flag[i]))
        elif (sim[i] < 0.7) or (not known_strain[i]):
            key = ("scaf", bool(std_flag[i]))
        else:
            key = ("known", bool(std_flag[i]))
        sig[i] = bundle["sigma_table"][key]

    # ── 信赖等级 ──
    levels, notes = [], []
    for i in range(n):
        if not targets[i]:
            levels.append("仅定性")
            notes.append("未指定菌株")
            continue
        bad = []
        if not known_strain[i]:
            bad.append("新菌株")
        if not known_type[i]:
            bad.append("新环化类型")
        if sim[i] < 0.7:
            bad.append("结构远离训练集")
        if not cond_known[i]:
            bad.append("条件未知")
        elif not std_flag[i]:
            bad.append("非标准条件")
        if not bad:
            levels.append("高")
            notes.append("菌株/环化/结构/条件均在训练范围内")
        elif len(bad) == 1:
            levels.append("中")
            notes.append("、".join(bad))
        else:
            levels.append("低")
            notes.append("、".join(bad))
    if calibrate:
        for i in range(n):
            if not known_type[i]:
                notes[i] += "；已按 k=0.51 收缩"

    out.loc[keep, "定量_log10uM"] = np.where(
        np.array([bool(t) for t in targets]), pred_cal, np.nan)
    out.loc[keep, "定量_uM"] = np.where(
        np.array([bool(t) for t in targets]), np.power(10.0, pred_cal), np.nan)
    lo = np.power(10.0, pred_cal - Z * sig)
    hi = np.power(10.0, pred_cal + Z * sig)
    has = np.array([bool(t) for t in targets])
    out.loc[keep, "定量_uM_low90"] = np.where(has, lo, np.nan)
    out.loc[keep, "定量_uM_high90"] = np.where(has, hi, np.nan)
    out.loc[keep, "σ_log10"] = np.where(has, sig, np.nan)
    out.loc[keep, "新颖度_菌株"] = ["已知" if k else "未知" for k in known_strain]
    out.loc[keep, "新颖度_环化"] = [
        ("已知" if k else "未知") for k in known_type]
    out.loc[keep, "环化类型"] = [t if t else "（未提供）" for t in types]
    out.loc[keep, "结构最大相似度"] = np.round(sim, 4)
    out.loc[keep, "条件是否标准"] = np.where(std_flag, "是", "否")
    out.loc[keep, "信赖等级"] = levels
    out.loc[keep, "说明"] = notes
    bad = out["状态"] != "成功"
    out.loc[bad, "说明"] = "SMILES 无法解析"
    out.loc[bad, "信赖等级"] = "—"

    # ── 定性通道 ──
    if with_qual:
        try:
            Xq = np.nan_to_num(X[:, :2277], nan=0.0, posinf=0.0, neginf=0.0)
            Pq, Sq, qtasks = _qual_predict(Xq, qual_bundle)
            cn = {"Any antibacterial": "总抗菌", "Antibacterial": "抗细菌",
                  "Gram-positive": "抗革兰阳性", "Gram-negative": "抗革兰阴性",
                  "Antifungal": "抗真菌"}
            cal = _load_cal().get("tasks", {})
            for j, t in enumerate(qtasks):
                name = cn.get(t, t)
                out.loc[keep, f"定性_{name}"] = np.round(Pq[:, j], 4)
                rel, grade = [], []
                for p_ in Pq[:, j]:
                    b = _cal_lookup(t, float(p_))
                    if b is None:
                        rel.append("—")
                        grade.append("无校准表")
                        continue
                    eq = "" if b.get("exact", True) else "≈"
                    rel.append(f"{eq}{b['obs_rate']:.3f} "
                               f"[{b['wilson_lo']:.3f},{b['wilson_hi']:.3f}] "
                               f"(n={b['n']})")
                    w = b["wilson_hi"] - b["wilson_lo"]
                    spans = b["wilson_lo"] < 0.5 < b["wilson_hi"]
                    grade.append(
                        "不确定（区间跨 0.5）" if spans else
                        "高" if w < 0.10 else
                        "中" if w < 0.25 else "低")
                out.loc[keep, f"定性_{name}_校准"] = rel
                out.loc[keep, f"定性_{name}_可靠度"] = grade
            # 逐目标判定：以「校准阳性率 ≥ 0.5」为准（可解释：更可能阳性）
            # 约登阈值仅为最大化 Sn+Sp 的作业点，多数远低于 0.5，
            # 用它判阳会导致几乎所有肽都被判阳性，故不用于最终判定。
            short = {"Any antibacterial": "总抗菌", "Antibacterial": "抗细菌",
                     "Gram-positive": "革兰阳性", "Gram-negative": "革兰阴性",
                     "Antifungal": "抗真菌"}
            order = ["总抗菌", "抗细菌", "革兰阳性", "革兰阴性", "抗真菌"]
            pos_flags, cal_rate = {}, {}
            for j, t in enumerate(qtasks):
                key = _TASK_EN.get(t, t)
                nm = short.get(key, key)
                rates, flags = [], []
                for p_ in Pq[:, j]:
                    b = _cal_lookup(t, float(p_))
                    r_ = 0.5 * float(p_) if b is None else float(b["obs_rate"])
                    rates.append(r_)
                    flags.append(r_ >= 0.5)
                cal_rate[nm] = np.array(rates)
                pos_flags[nm] = np.array(flags)
            if pos_flags:
                decide, grade = [], []
                for i in range(len(Pq)):
                    # 生物学约束：总抗菌 = 抗细菌 ∪ 革兰阳性 ∪ 革兰阴性
                    any_pos = any(pos_flags[n][i] for n in
                                  ("抗细菌", "革兰阳性", "革兰阴性")
                                  if n in pos_flags)
                    if "总抗菌" in pos_flags:
                        pos_flags["总抗菌"][i] = bool(
                            pos_flags["总抗菌"][i] or any_pos)
                    hit = [n for n in order if n in pos_flags and pos_flags[n][i]]
                    decide.append("阴性" if not hit else "阳性：" + "、".join(hit))
                    if not hit:
                        grade.append("阴性")
                    elif "总抗菌" in hit:
                        grade.append("广谱" if len(hit) >= 3 else "窄谱")
                    else:
                        grade.append("仅特定靶标")
                out.loc[keep, "判定"] = decide
                out.loc[keep, "判定类型"] = grade
                for nm in order:
                    if nm in cal_rate:
                        out.loc[keep, f"定性_{nm}_校准阳性率"] = np.round(
                            cal_rate[nm], 3)

        except Exception as e:  # pragma: no cover
            print(f"  [warn] 定性通道不可用（{type(e).__name__}: {e}）",
                  file=sys.stderr)

    # ── 按菌株训练支撑度修正信赖等级 ──
    sup = _strain_support()
    if sup and "信赖等级" in out.columns:
        grades, notes = list(out["信赖等级"]), list(out.get("说明", [""] * len(out)))
        for pos, i in enumerate(keep):
            st = targets[pos] if pos < len(targets) else ""
            info = sup.get(st)
            if not info:
                continue
            lv, r2 = info.get("level"), info.get("R2")
            add, why = 0, ""
            if r2 is not None and r2 < 0:
                add, why = 2, f"该菌株 OOF R²={r2:+.2f}（不如取均值），仅供定性参考"
            elif lv == "弱":
                add, why = 1, f"该菌株仅 {info['n']} 对样本、R²={r2:+.2f}，把握偏低"
            elif lv == "可用":
                add, why = 0, f"该菌株 {info['n']} 对样本、R²={r2:+.2f}"
            if add:
                grades[i] = _downgrade(grades[i], add)
            if why:
                notes[i] = (str(notes[i]) + "；" if notes[i] else "") + why
        out["信赖等级"] = grades
        out["说明"] = notes

    if with_qual:
        out.loc[keep, "区间说明"] = (
            "定量 σ 区间 = 预测误差尺度；定性「校准」= 该概率分箱的实测阳性率"
            "及 Wilson 95% 区间，两者含义不同，不可互换")
    out.loc[keep, "模型版本"] = bundle["version"]

    if use_esm:
        out.loc[keep, "ESM融合"] = ("已融合 w=%.2f" % bundle["w_esm"]
                                    if esm_used == n else
                                    f"{esm_used}/{n} 条融合，其余仅主模型")
    if verbose:
        print(f"  [v2] n={n}  主模型 {XM.shape}  "
              f"ESM={'on' if use_esm else 'off'}  融合 {esm_used}/{n}  "
              f"w={bundle['w_esm'] if esm_used else 0}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    r = predict(df, verbose=True)
    cols = [c for c in ("SMILES", "Strain", "cfuGroup", "Medium",
                        "CyclizationType", "定量_uM", "定量_uM_low90",
                        "定量_uM_high90", "σ_log10", "信赖等级", "说明")
            if c in r.columns]
    print(r[cols].to_string(index=False))
    if a.out:
        r.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"\n  → {a.out}")


if __name__ == "__main__":
    main()


def to_compact(r):
    """把完整结果压成默认展示的 10 列。"""
    keep = []
    if "ID" in r.columns:
        keep.append("ID")
    keep.append("定量_uM")
    if {"定量_uM_low90", "定量_uM_high90"} <= set(r.columns):
        r = r.copy()
        r["定量区间"] = [
            "—" if (pd.isna(lo) or pd.isna(hi))
            else f"{lo:.3g}–{hi:.3g}"
            for lo, hi in zip(r["定量_uM_low90"], r["定量_uM_high90"])]
        keep.append("定量区间")
    if "信赖等级" in r.columns:
        keep.append("信赖等级")
    for t in ("总抗菌", "抗细菌", "抗革兰阳性", "抗革兰阴性", "抗真菌"):
        c = f"定性_{t}"
        c2 = f"定性_{t}(H2)"
        if c in r.columns:
            keep.append(c)
        elif c2 in r.columns:
            keep.append(c2)
    if "判定" in r.columns:
        keep.append("判定")
    cols = [c for c in keep if c in r.columns]
    out = r[cols].copy()
    ren = {c: c.replace("(H2)", "") for c in cols}
    return out.rename(columns=ren)


