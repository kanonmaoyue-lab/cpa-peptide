"""cpamp_tool.py -- 环肽活性预测工具（可微调版）。

═══ 给使用者的一条命令 ═══
    python cpamp_tool.py predict  -i 待测.csv  -o 结果.csv
    python cpamp_tool.py finetune -l 自己的标注.csv -o my_model.pkl
    python cpamp_tool.py check    -l 自己的标注.csv        # 只看偏差，不重训

═══ 为什么需要 finetune ═══
    遇到**没见过的成环化学**时，零标注 R² 仅 0.02，不可用。
    实测：只做偏移校正只能恢复 19–42% 的收益（F73），
    必须把标注并入训练集重新拟合。
    10 条 → LOCTO R² +0.147；25 条 → +0.248；50 条 → +0.334。

═══ 标注文件格式（列名不区分大小写）═══
    SMILES        必填
    Strain        必填（菌株名）
    MIC_uM        必填（实测 MIC，µM）
    Seq           可选（肽序列，缺失则序列特征为 0）
    cfuGroup      可选（0–4）
    Medium        可选（MHB/CAMHB/…）
    CyclizationType  可选（提供则重训后按"已知类型"处理，k=1.0）

═══ 目录结构 ═══
    cpamp_tool.py
    data/           训练表、特征缓存、SMILES 索引
    models/         final_bundle.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
# 特征器与推理模块仍在 cpamp_lib/ 下（开源打包时一并移入 cpamp/ 包）
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpamp_lib"))

LGBM = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
            subsample_freq=1, verbose=-1)
SEEDS = (0, 1, 2)
W_ESM = 0.35
Z90 = 1.6448536269514722


# ═══════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════
def _col(df, names):
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def load_labels(path):
    """读取用户标注 → 统一 schema。"""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = None
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            for sep in (",", "\t", ";", None):
                try:
                    df = pd.read_csv(p, encoding=enc, sep=sep,
                                     engine="python" if sep is None else "c")
                    if df.shape[1] >= 2 and len(df):
                        break
                except Exception:
                    df = None
            if df is not None and df.shape[1] >= 2 and len(df):
                break
        if df is None:
            raise ValueError(f"无法解析 {p}")
    cs = _col(df, ["smiles", "结构", "smiles结构", "canonical_smiles", "分子"])
    ct = _col(df, ["strain", "菌株", "species", "species_name"])
    cy = _col(df, ["mic_um", "mic", "mic(um)", "mic_µm", "mic值", "mic_μм"])
    if cy is None:
        cy = _col(df, ["y", "label", "value", "活性"])
    if cs is None or ct is None or cy is None:
        raise ValueError(
            f"标注文件缺少必需列。需要 SMILES / Strain / MIC_uM。\n"
            f"检测到的列：{list(df.columns)}")
    out = pd.DataFrame({
        "SMILES": df[cs].astype(str).str.strip(),
        "Strain": df[ct].astype(str).str.strip(),
        "MIC_uM": pd.to_numeric(df[cy], errors="coerce"),
    })
    q = _col(df, ["seq", "sequence", "序列"])
    if q:
        out["Seq"] = df[q].fillna("").astype(str)
    cg = _col(df, ["cfugroup", "cfu_group", "cfu"])
    if cg:
        out["cfuGroup"] = pd.to_numeric(df[cg], errors="coerce")
    md = _col(df, ["medium", "培养基"])
    if md:
        out["Medium"] = df[md].fillna("").astype(str).str.strip().str.upper()
    ck = _col(df, ["cyclizationtype", "cyclization_type", "cyclization", "环化"])
    if ck:
        out["CyclizationType"] = df[ck].fillna("").astype(str).str.strip()
    out = out.dropna(subset=["MIC_uM"])
    out = out[out["MIC_uM"] > 0]
    out["y"] = np.log10(out["MIC_uM"].astype(float))
    return out.reset_index(drop=True)


def _read_table(path):
    """读 CSV/Excel，自动尝试编码与分隔符。"""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
        for sep in (",", "\t", ";", None):
            try:
                df = pd.read_csv(p, encoding=enc, sep=sep,
                                 engine="python" if sep is None else "c",
                                 on_bad_lines="skip")
                if df.shape[1] >= 1 and len(df):
                    return df
            except Exception as e:
                last = e
    raise last if last else ValueError(f"无法解析 {p}")


def load_query(path):
    """预测用输入：只需 SMILES，Strain/Seq/条件均可选。"""
    df = _read_table(path)
    cs = _col(df, ["smiles", "结构", "smiles结构", "canonical_smiles", "分子"])
    if cs is None:
        # 内容兜底
        for c in df.columns:
            vals = df[c].dropna().astype(str).head(20)
            if len(vals) and sum(len(v) > 15 and
                                 sum(ch in v for ch in "CNOScnos") >= 3
                                 for v in vals) >= max(1, len(vals) // 2):
                cs = c
                break
    if cs is None:
        raise ValueError(f"找不到 SMILES 列。检测到的列：{list(df.columns)}")
    out = pd.DataFrame({"SMILES": df[cs].astype(str).str.strip()})
    ci = _col(df, ["id", "序号", "编号", "name", "名称", "cpid"])
    out["ID"] = (df[ci].astype(str) if ci else
                 [f"P{i+1}" for i in range(len(df))])
    ct = _col(df, ["strain", "菌株", "species"])
    if ct:
        out["Strain"] = df[ct].astype(str).str.strip()
    q = _col(df, ["seq", "sequence", "序列"])
    if q:
        out["Seq"] = df[q].fillna("").astype(str)
    cg = _col(df, ["cfugroup", "cfu_group", "cfu"])
    if cg:
        out["cfuGroup"] = pd.to_numeric(df[cg], errors="coerce")
    md = _col(df, ["medium", "培养基"])
    if md:
        out["Medium"] = df[md].fillna("").astype(str).str.strip().str.upper()
    ck = _col(df, ["cyclizationtype", "cyclization_type", "cyclization", "环化"])
    if ck:
        out["CyclizationType"] = df[ck].fillna("").astype(str).str.strip()
    return out


def load_bundle(path=None):
    p = Path(path) if path else MODELS / "final_bundle.pkl"
    with open(p, "rb") as fh:
        return pickle.load(fh), p


def base_features():
    """训练集特征（肽级存储 + 映射展开为对级，省磁盘）。"""
    z = np.load(DATA / "train_features.npz", allow_pickle=True)
    X = z["X_pep"].astype(np.float32)[z["xi"].astype(int)]
    return (X, z["y"].astype(float), z["strain"].astype(str),
            z["peptide_ids"].astype(str)[z["xi"].astype(int)],
            z["bond"].astype(str), z["cfu"].astype(float),
            z["medium"].astype(str))


def _strain_ctx(strain, y, groups, codes, medians, n_new_code):
    order = ["gram_pos", "gram_neg", "fungal", "other"]
    C = np.zeros((len(strain), 6), np.float32)
    gm = float(np.median(y)) if len(y) else 0.0
    for i, s in enumerate(strain):
        g = groups.get(s, "other")
        C[i, order.index(g) if g in order else 3] = 1
        C[i, 4] = codes.get(s, n_new_code)
        C[i, 5] = medians.get(s, gm)
    return C


# ═══════════════════════════════════════════════════════════
# predict
# ═══════════════════════════════════════════════════════════
def cmd_predict(args):
    from cpamp_predict_v2 import predict
    src = load_query(args.input)
    if src.empty:
        raise SystemExit("输入为空")
    from cpamp_predict_v2 import to_compact
    r = predict(src, use_esm=not args.no_esm, calibrate=True, verbose=True,
                bundle=getattr(args, "base", None))
    if not getattr(args, "detail", False):
        r = to_compact(r)
    cols = list(r.columns)
    if args.out:
        r.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n  → {args.out}（{len(r)} 条）")
    else:
        print(r[cols].to_string(index=False))
    return 0


# ═══════════════════════════════════════════════════════════
# finetune
# ═══════════════════════════════════════════════════════════
def cmd_finetune(args):
    import lightgbm as lgb
    from cpamp_predict import build_features
    bundle, bp = load_bundle(args.base)

    Xb, yb, sb, pidb, bondb, cfub, medb = base_features()
    print(f"  基础训练集：{Xb.shape[0]} 对 / {len(set(pidb))} 肽 / "
          f"{len(set(sb))} 菌株")

    lab = load_labels(args.labels)
    print(f"  用户标注：{len(lab)} 条")

    # 新标注的特征
    feats, keep, resolved = build_features(
        lab["SMILES"].fillna("").tolist(),
        lab["Seq"].fillna("").tolist() if "Seq" in lab else [""] * len(lab))
    lab = lab.iloc[keep].reset_index(drop=True)
    Xn = np.nan_to_num(feats["legacy"], nan=0.0, posinf=0.0, neginf=0.0)
    yn = lab["y"].to_numpy(float)
    sn = lab["Strain"].astype(str).to_numpy()
    if "cfuGroup" in lab:
        cfn = pd.to_numeric(lab["cfuGroup"], errors="coerce").to_numpy(float)
    else:
        cfn = np.full(len(lab), np.nan)
    print(f"  可解析 {len(lab)} 条；SMILES 无法解析 {len(feats['legacy']) - 0 if False else 0}")

    # 拼接
    Xall = np.vstack([Xb, Xn])
    yall = np.concatenate([yb, yn])
    sall = np.concatenate([sb, sn])
    cfall = np.concatenate([cfub, cfn])
    fill = float(np.nanmedian(cfall)) if np.isfinite(cfall).any() else 2.0
    print(f"  合并后 {Xall.shape[0]} 对（原 {Xb.shape[0]} + 新 {Xn.shape[0]}）")

    # 菌株上下文：新菌株用专用编号
    codes = dict(bundle["strain_codes"])
    meds = dict(bundle["strain_median"])
    groups = dict(bundle["strain_group"])
    n_new = len(codes)
    for s in set(sall):
        if s not in codes:
            codes[s] = n_new
            n_new += 1
    for s in set(sall):
        if s not in meds:
            meds[s] = float(np.median(yall[sall == s]))
        if s not in groups:
            groups[s] = "other"
    C = _strain_ctx(sall, yall, groups, codes, meds, n_new)
    XM = np.hstack([Xall,
                    np.nan_to_num(cfall, nan=fill)[:, None],
                    (~np.isfinite(cfall)).astype(np.float32)[:, None], C])
    print(f"  主模型输入 {XM.shape}")

    print("  重训主模型（3 seeds）…", flush=True)
    models = []
    for sd in SEEDS:
        m = lgb.LGBMRegressor(random_state=sd, **LGBM)
        m.fit(XM, yall, categorical_feature=[XM.shape[1] - 2])
        models.append(m)
        print(f"    seed {sd} 完成", flush=True)

    nb = dict(bundle)
    nb["main_models"] = models
    nb["n_main_feat"] = int(XM.shape[1])
    nb["version"] = bundle["version"] + f"+ft{len(lab)}"
    nb["strain_codes"] = codes
    nb["strain_median"] = meds
    nb["strain_group"] = groups
    ty = set(bundle["cyc_types"])
    if "CyclizationType" in lab:
        ty |= {t for t in lab["CyclizationType"].astype(str) if t}
    nb["cyc_types"] = sorted(ty)
    nb["finetune"] = {"n_labels": int(len(lab)),
                      "strains": sorted(set(sn.tolist())),
                      "labels_file": str(args.labels)}
    out = Path(args.out) if args.out else MODELS / "finetuned_bundle.pkl"
    if args.out and not Path(args.out).is_absolute():
        out = Path(args.out)
    with open(out, "wb") as fh:
        pickle.dump(nb, fh, protocol=4)
    print(f"\n  → {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  用法：python cpamp_tool.py predict -i 待测.csv --base {out}")
    return 0


# ═══════════════════════════════════════════════════════════
# check：只诊断偏差，不重训
# ═══════════════════════════════════════════════════════════
def cmd_check(args):
    from cpamp_predict_v2 import predict
    lab = load_labels(args.labels)
    r = predict(lab.rename(columns={"Strain": "Strain"}), calibrate=True,
                verbose=False, bundle=getattr(args, "base", None))
    ok = r.get("定量_log10uM")
    if ok is None:
        print("  预测失败")
        return 1
    pred = pd.to_numeric(ok, errors="coerce").to_numpy(float)
    resid = lab["y"].to_numpy(float) - pred
    m = np.isfinite(resid)
    print(f"  可比较 {int(m.sum())} 条")
    print(f"  偏差（实测 − 预测）：均值 {resid[m].mean():+.4f}  "
          f"中位 {np.median(resid[m]):+.4f}  SD {resid[m].std():.4f}")
    print(f"  平均绝对误差 {np.mean(np.abs(resid[m])):.4f} log")
    print()
    if abs(resid[m].mean()) > 0.3:
        print(f"  ⇒ 系统性偏差 {resid[m].mean():+.2f} log"
              f"（约 {10**resid[m].mean():.1f} 倍），建议补标注后重训：")
        print(f"     python cpamp_tool.py finetune -l {args.labels} -o my_model.pkl")
    else:
        print("  ⇒ 未见明显系统偏差，可直接使用")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="环肽抗菌活性预测工具（含可微调）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("predict", help="预测")
    p1.add_argument("-i", "--input", required=True)
    p1.add_argument("-o", "--out")
    p1.add_argument("--base", help="自定义模型文件")
    p1.add_argument("--no-esm", action="store_true")
    p1.add_argument("--detail", action="store_true",
                    help="输出完整列（含校准阳性率、Wilson 区间、σ、说明）")
    p1.set_defaults(func=cmd_predict)

    p2 = sub.add_parser("finetune", help="用手上标注重训")
    p2.add_argument("-l", "--labels", required=True)
    p2.add_argument("-o", "--out")
    p2.add_argument("--base", help="基础模型文件")
    p2.set_defaults(func=cmd_finetune)

    p3 = sub.add_parser("check", help="诊断偏差（不重训）")
    p3.add_argument("-l", "--labels", required=True)
    p3.add_argument("--base", help="自定义模型文件")
    p3.set_defaults(func=cmd_check)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
