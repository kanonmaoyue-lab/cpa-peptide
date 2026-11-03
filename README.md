# 双通道环肽抗菌活性预测

一个可标注学习的环肽 MIC 预测系统：用分子结构预测抗菌活性，遇到训练集里没有的成环化学时，可用少量自有标注重训，把该类型从"不可用"变成"可用"。

---

## 1 功能

| 命令 | 用途 | 耗时 |
|---|---|---|
| `predict` | 输入 SMILES，输出 MIC 点预测 + 90% 区间 + 信赖等级 | 秒级 |
| `check` | 用已有实测数据诊断模型偏差 | 秒级 |
| `finetune` | 把标注并入训练集**重训**，生成专属模型 | 3–8 分钟 |

图形界面（`start.bat`）提供同样的三项功能，并带菌株支撑度提示。

---

## 2 性能

**定量通道**（同分布，n = 14 453 对）

| 指标 | 值 |
|---|---|
| R² | **0.6616** |
| MAE | 0.3765 log₁₀ |
| Pearson / Spearman | 0.8139 / 0.7878 |
| ±1 log 命中率 | 93.2% |

**定性通道**（5 类活性）：平均 AUROC **0.9427**

**六协议梯度**

| 协议 | R² |
|---|---:|
| 同分布肽簇 5 折 | 0.6616 |
| 未见骨架 | 0.5281 |
| 未见成环化学 | 0.0215 |
| 跨菌株外部（零先验） | 0.6339 |
| 留一文献 | 0.4629 |
| 前瞻时间切分 | **0.3201**（最接近真实部署）|

跨菌株外部口径（0.6339）与同类公开模型 tsAMP（0.634）持平。

---

## 3 核心特性

**① 条件信息比表征复杂度更重要**
6 维菌株上下文带来 +0.2084 的 R² 提升；而把 2 048 维指纹扩到 2 331 维多视图只值 +0.0233。单块 Morgan 指纹单独用已优于全部特征。

**② 融合方式本身是独立变量**
同一 ESM-2 嵌入，直接拼接为 −0.0094、预测级融合为 +0.0128。机制是列采样下稠密连续特征稀释了离散指纹。

**③ 双通道输出，两种不确定度不通用**
定量给 MIC 的 90% 区间（预测误差尺度）；定性给每类活性的校准阳性率与 Wilson 95% 区间（概率的可靠性）。二者含义不同，不可互换。

**④ 标注学习必须重训，不能在推理端平移**
实测"只估偏移量再加回去"只能恢复 19–42% 的收益。

| 新增标注 | 该类型 R² |
|---:|---:|
| 0 | 0.0215（不可用）|
| 10 | 0.147 |
| 25 | 0.248 |
| 50 | 0.334 |

**⑤ cfu 字段的作用限于插值**
补入接种量字段后同分布 +0.0040、未见骨架 +0.0028，但未见成环化学 −0.0022。它的价值在适用域与区间校准，不在精度。

---

## 4 安装

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt        # Linux / macOS
```

核心依赖：`numpy pandas scipy scikit-learn lightgbm rdkit`

ESM 序列分支为可选项（`torch` + `transformers`）。不装则自动回退为仅主模型，R² 约降 0.013，定量结果仍然有效。

> ⚠️ 模型文件 `models/final_bundle.pkl` 由 **Git LFS** 托管。克隆后若该文件只有 130 字节，执行 `git lfs pull`。

---

## 5 快速开始

**预测**

```bash
python cpamp_tool.py predict -i my_peptides.csv -o result.csv
```

输入最少只需两列：`ID, SMILES`。加上 `Strain / cfuGroup / Medium / CyclizationType` 可缩小预测区间、提高信赖等级。

**输出**（默认精简 10 列）

`ID, 定量_uM, 定量区间下限/上限, 定性各目标概率, 判定`

判定规则：校准阳性率 ≥ 0.5 判为阳性，并满足生物学约束（总抗菌 = 抗细菌 ∪ 革兰阳性 ∪ 革兰阴性）。

**诊断偏差**

```bash
python cpamp_tool.py check -l my_labels.csv
```

**标注学习**

```bash
python cpamp_tool.py finetune -l my_labels.csv -o my_model.pkl
python cpamp_tool.py predict -i new.csv --base my_model.pkl
```

标注文件至少需要：`SMILES, Strain, MIC_uM`（可选 `CyclizationType`）。

---

## 6 适用域与局限

**可以用**：已知化学型 + 已知或同组菌株的活性定量；标准肉汤微量稀释条件（MHB/CAMHB，10⁴–10⁷ CFU/mL）内 R² ≈ 0.69、±1 log 94.3%。

**不适用**：

| 情形 | 表现 |
|---|---|
| 全新成环化学、零标注 | R² ≈ 0.02，即使最优校准也仅 0.04 → 需 ≥10 条标注重训 |
| 全新肽骨架 | R² = −0.1127，无支撑 |
| 分档决策（敏感/耐药）| 不建议，应使用区间而非单点 |

**菌株支撑度**：覆盖 155 株，但

| 判定 | 株数 | 占比 |
|---|---:|---:|
| 充分（R² ≥ 0.4）| 81 | 52.3% |
| 可用（R² ≥ 0.2）| 36 | 23.2% |
| 弱 | 38 | 24.5% |
| 其中 R² < 0 | 18 | 11.6% |

对 R² < 0 的菌株，系统自动把信赖等级下调两级并注明"仅供定性参考"。
详见 [docs/strain_audit.md](docs/strain_audit.md)。

---

## 7 模型结构

| 通道 | 输入 | 模型 | 融合 |
|---|---|---|---|
| 定量 | 2 331 描述符 + cfu 2 + 菌株上下文 6 = **2 339 维** | LightGBM × 3 seeds | 与冻结 ESM-2（960 维）预测级融合，w = 0.35 |
| 定性 | 2 277 描述符 + 目标编码 5 = **2 282 维** | LightGBM 统一模型（1 个替代原 5 个）| w = 0.50 |

LightGBM 超参：600 棵树、63 叶、学习率 0.05、最小叶样本 15、列采样 0.7、行采样 0.8。

**不确定度**：90% 区间 = `10^(ŷ ± 1.645 σ)`，σ 由「新颖性 × 测定条件」两轴查表给出（同分布 0.49 / 新骨架 0.60 / 新成环 0.87）。

---

## 8 文件索引

```
cpamp_tool.py                命令行入口（predict / check / finetune）
cpamp_gui.py                 图形界面
start.bat / start.sh         一键启动
requirements.txt             依赖

cpamp_lib/                   推理链的 14 个模块
  90_build_expanded.py       2 331 维分子特征计算
  03_features.py             Morgan 指纹与理化描述符
  26/27/30_*.py              环化相关的特征与清洗
  cpamp_predict.py           双通道推理
  cpamp_predict_v2.py        本版推理：cfu + ESM 融合 + 区间 + 适用域
  cpamp_retrain.py           重训工具（供 finetune 使用）
  cpamp_sequence.py          序列处理
  13/14/15/69/98_*.py        特征与模型的辅助模块

data/
  train_features.npz         训练特征（微调必需）
  train_table.csv            训练表（14 453 行）
  strains.csv                155 株清单
  cyc_types.csv              18 种环化类型

datasets/DBAASP_full/
  quant_dataset.pkl          特征器依赖的列定义
  expanded.npz               2 331 个特征名

artifacts/
  qual_calibration.json      定性通道校准表
  esm2/esm2_t12_emb.npz      ESM-2 嵌入缓存（3 094 条序列）
  pantarget/pantarget_models.pkl   定性统一模型

models/
  final_bundle.pkl           打包模型（3+3 个 LightGBM、σ 表、菌株支撑度）

assets/                      界面素材
examples/                    测试数据（3 个 CSV，从场景演示到仅 SMILES）
docs/                        菌株审计、部署说明、实验记录
```

测试数据说明见 [docs/demo_guide.md](docs/demo_guide.md)。

---

## 9 引用与许可

代码以 MIT 许可发布，见 [LICENSE](LICENSE)。

模型基于 DBAASP v4.0.1 与 CyclicPepedia v1.3.1 构建。使用本工具发表成果时，请同时引用这两个数据库：

- DBAASP v4.0.1 — <https://dbaasp.org>
- CyclicPepedia v1.3.1 — <https://cycpeptmpdb.com>

本工作的方法学论文正在准备中。
