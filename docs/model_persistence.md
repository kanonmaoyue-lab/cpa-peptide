# 模型持久化（重训后保存与复用）

## 为什么

标注学习（finetune）生成的模型原本只存在于当次会话，关掉窗口就丢失，
下次打开又要重新导入标注、重新训练。

本机制让重训结果**保存下来**，下次打开直接可用，并可随时回退。

## 怎么工作

`
models/finetuned_20261117_214634.pkl    ← 每次重训独立文件，不覆盖
models/active_model.txt                 ← 一行文本：当前生效的模型名
models/final_bundle.pkl                 ← 基础模型（仓库自带）
`

启动时解析顺序：

1. ctive_model.txt 存在，且它指向的 .pkl 也存在 → 用该模型
2. 否则 → models/final_bundle.pkl

记录里只存**文件名**，不存绝对路径，换目录不会失效。

## 界面

窗口右上角是模型下拉条，按修改时间倒序列出 models/*.pkl：

`
finetuned_20261117_214634.pkl  ·  11-17 21:46  ·  54MB
finetuned_bundle.pkl           ·  11-17 20:56  ·  54MB
final_bundle.pkl               ·  基础模型      ·  54MB
`

- 在「标注学习」页完成一次重训后，新模型自动保存、自动选中，提示
  「已保存并切换到 …（重启后仍生效）」
- 在下拉条里选任意一项即立即生效并记住
- 关闭窗口再打开，仍停在上次选择

## 命令行

`
python cpamp_tool.py predict -i new.csv            # 用当前模型
python cpamp_tool.py predict -i new.csv --base models/final_bundle.pkl   # 显式指定
`

## 注意

- inetuned_*.pkl 与 ctive_model.txt 已加入 .gitignore，属于本地状态，不随仓库分发
- 旧模型不会自动清理；在 models/ 里删掉对应 .pkl 后，下拉列表里自然消失
- 记录指向的文件若被删除，启动时静默回退到基础模型