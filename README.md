# RT 视觉组传统视觉练习

对应飞书《传统视觉》第五节。四项练习共用一段篮球视频，各自可以独立运行。作业要求和验收标准见培训文档。

[《传统视觉》培训任务](https://gcnhkq914dqv.feishu.cn/wiki/YlfNwQq6miKhxdki6FFcHN1qn0c#doxcnt824HA8qzh9FQ62jNlX0ce)

## 开始

Fork 本仓库，克隆自己的 Fork，并按《开发管理》的规范创建个人工作分支。以下命令在仓库根目录执行，适用于 Ubuntu 或 WSL，使用 Python 3.10–3.13。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

在 Ubuntu 上，如果创建虚拟环境时提示缺少 `ensurepip`，安装 `python3-venv` 后重新创建。

## 练习入口

| 练习 | 命令 | 输出 |
| --- | --- | --- |
| 任务一 · 选择图像特征 | `python -m vision_lab task1` | `runs/task1/frame-100.png` |
| 任务二 · 定位篮球漏检 | `python -m vision_lab task2` | `runs/task2/` 下的两张阶段图及 `candidates.csv` |
| 任务三 · 评估闭运算的效果 | `python -m vision_lab task3` | `runs/task3/frame-150.png` |
| 任务四 · 验证形状阈值改动 | `python -m vision_lab task4` | `runs/task4/` 下的逐帧对照图、`predictions.csv` 和空白结果表 `submission.csv` |

- 运行时默认读取仓库自带的 `samples/basketball.mp4`。帧号从 0 开始。
- 图中的 `found` 表示程序输出了候选位置，`none` 表示没有输出。任务二的候选编号与 `candidates.csv` 对应。
- 任务四的 `predictions.csv` 记录程序输出，`submission.csv` 用于填写人工核对结果。重新运行会保留已有的 `submission.csv`。

## 提交

按飞书任务要求提交图、表和判断。在自己的工作分支里保存需要提交的文件，可以放在 `submissions/task1` 至 `submissions/task4`。默认输出目录 `runs/` 已被 Git 忽略。推送后，问卷填写工作分支直达链接，并按题提交材料。

样例视频和基线检测规则来自 [篮球追踪输出练习](https://github.com/Ecrypted-Data/rt-basketball-output-lab/tree/0d3c5830d7a43fc247a824657068524f6153d103)。
