# Kaiwu SDK 环境与使用说明

## 环境位置
- **conda env 名**：`kaiwu-py310`
- **Python**：3.10
- **物理路径**：`D:\Anaconda\envs\kaiwu-py310\`
- **专属用途**：仅跑 Kaiwu SDK / 真机调用、构造与提交 QUBO。日常分析、出图仍在原 Python 3.13 环境。

## 激活方式
```bash
# Windows bash / git bash
source D:/Anaconda/etc/profile.d/conda.sh
conda activate kaiwu-py310

# 或 PowerShell
conda activate kaiwu-py310
```

## 直接调用解释器（无需激活）
```
D:/Anaconda/envs/kaiwu-py310/python.exe  src/02_q1_kaiwu_solve.py
```

## 已安装包（基础）
- numpy / pandas / openpyxl / matplotlib
- kaiwu-sdk（待安装，等用户领取算力配额并提供 token / pip 源后再装）

## 安装 Kaiwu SDK
官方下载入口：<https://platform.qboson.com/sdkDownload>
教程：<https://kaiwu.qboson.com/plugin.php?id=knowledge>

安装命令（示例，具体以官方为准）：
```bash
D:/Anaconda/envs/kaiwu-py310/python.exe -m pip install kaiwu-sdk
# 或离线 wheel：
D:/Anaconda/envs/kaiwu-py310/python.exe -m pip install path/to/kaiwu_sdk-*.whl
```

## 真机算力 token
扫码领取后，token 配置方式（具体以 SDK 文档为准）：
```python
import kaiwu as kw
kw.cim.set_token("YOUR_TOKEN_HERE")
```

## 流程图
```
build_qubo() → Q (225x225 numpy)
       ↓
[Python 3.13]: 构造 Q、解码、2-opt、出图（不变）
       ↓
[Python 3.10 + Kaiwu SDK]: kw.solver.SimulatedAnnealingOptimizer / CIMSolver 求解
       ↓
返回 0/1 解 → 解码 → 总时间
```
