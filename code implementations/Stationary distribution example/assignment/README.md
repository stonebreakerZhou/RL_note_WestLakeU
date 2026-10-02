# Stationary Distribution 演示（2×2 网格）

> 对应笔记：`RL_note/RL_note_WestLakeU/Sean's notes/RL_WLU_note.typ` §"Stationary distribution"（约第 4130 行）。

---

## 一、题目背景

**stationary distribution** `d_π(s)` 描述的是：在**给定策略 π** 下，agent 跑了**很久**之后，处于状态 `s` 的概率（长程稳态分布）。

笔记原文：
> *"It describes the long-run behavior of a Markov process. By definition, d_π(s) ≥ 0 and Σ d_π(s) = 1."*

`d_π` 由下面的不动点方程决定：
```
d_π = d_π · P_π           (P_π 是跟着 π 走的转移矩阵)
```

所以"经验占比"在 t 足够大时会收敛到 `d_π`：
```
lim  (#steps at s) / (t)   =   d_π(s)
t→∞
```

---

## 二、本示例要做什么

**故意做得极简**——2×2 网格（4 个状态），用一个**手动给定的策略 π**（不是学出来的、不是 ε-greedy），agent 从一个起点出发跑 1000 步，画出**四条曲线**：

- **横轴**：step index（0 → 1000）
- **纵轴**：Percentage of each state visited（也就是 `count(s) / t`）
- **4 条实线**：4 个状态的经验占比，用不同颜色区分

学生应该看到：曲线从震荡逐渐**收敛**到某个稳定值——这就是"长程稳态"的含义。理论 `d_π(s)` 的数值会在**终端打印**（不画在图上，保持画面简洁）。

---

## 三、环境与参数

| 项 | 值 | 说明 |
|----|----|------|
| 网格 | 2×2 | 4 个状态 `(0,0)`、`(1,0)`、`(0,1)`、`(1,1)` |
| 动作 | 5 个：`down / right / up / left / stay` | 跟 Sarsa/Q-learning 一致 |
| 转移 | **确定性**，越界 → bump → 原地不动 | 跟其他例子一致 |
| 策略 `π` | **`PI` 矩阵**，4×5 | **手动调**（每行加起来必须 = 1）|
| 起点 | `(0,0)` | `START_STATE = 0` |
| 步数 | 1000 | `NUM_STEPS = 1000` |

**唯一需要手动调的常量**：`PI`（`train.py` 顶部）。其他都标了 `# 可调`，按需改。

---

## 四、操作步骤

```bash
# 1. 进入目录
cd "E:/RL-西湖大学赵世钰/RL-WestlakeU-Zhao-main/RL_note/RL_note_WestLakeU/code implementations/Stationary distribution example/assignment"

# 2. 装依赖（只有 numpy + matplotlib）
pip install -r requirements.txt

# 3. 跑默认策略
python train.py
```

会弹出**一张图**（4 条经验曲线），控制台会打出 `d_π` 的理论值和 1000 步后的经验占比。

---

## 五、怎么调 `PI`（手动改策略看效果）

打开 `train.py` 顶部，找到这块：

```python
PI = np.array([
    # state 0 (0,0): mostly stay + drift right
    [0.10, 0.40, 0.00, 0.00, 0.50],
    # state 1 (1,0): down + left + stay
    [0.30, 0.00, 0.00, 0.30, 0.40],
    # state 2 (0,1): mostly stay + drift right
    [0.00, 0.40, 0.10, 0.00, 0.50],
    # state 3 (1,1): up + left + stay
    [0.00, 0.00, 0.30, 0.30, 0.40],
])
```

- 行 = 4 个状态（顺序见 `STATES` 列表）
- 列 = 5 个动作（顺序：down / right / up / left / stay）
- **每一行加起来必须正好等于 1**（脚本里有 `assert`，行不满足会报错）

**几个值得试的修改方向**：

1. **改成"均匀策略"** —— 把每行都设成 `[0.2, 0.2, 0.2, 0.2, 0.2]`。理论 `d_π` 会变成 `(0.25, 0.25, 0.25, 0.25)`，四条曲线**全收敛到 0.25**。最对称、也最无聊。
2. **让某个状态"粘住"** —— 把某行设成 `[0, 0, 0, 0, 1]`（永远 stay）。那个状态变成吸收态，`d_π` 在那一点等于从起点到它的概率。
3. **偏向某个角落** —— 让 `(0,0)` 行大概率往右 (`[0, 0.8, 0, 0, 0.2]`)，让 `(1,0)` 行大概率往左 (`[0, 0, 0, 0.8, 0.2]`)——形成左右"震荡"，`d_π((0,0)) = d_π((1,0)) = 0.5`。
4. **打破对称** —— 让 `(0,0)` 和 `(0,1)` 都偏 stay（`d_π` 在上面两个状态变大）；让 `(1,0)` 和 `(1,1)` 都偏移动——上下分布不均。

每改一次 `PI`，重新跑 `python train.py`，看四条曲线收敛到的新位置。

---

## 六、期望看到什么

### 默认策略（带轻微"向下/向右"漂移）

控制台会打印（理论）：
```
Stationary distribution d_π (theoretical):
  (0, 0):  0.2143
  (1, 0):  0.2857
  (0, 1):  0.2143
  (1, 1):  0.2857
```

注意 `(1,0)` 和 `(1,1)` 比上面两个状态高一点点——因为默认策略在下半部分比上半部分"移动得更活跃"（见 `PI` 的后两行），停留概率 0.4 比上半部分的 0.5 小。

### 图上你应该看到

- 一条**蓝色实线**（`(0,0)` 的经验占比）：从 1.0 开始（前几步几乎只在起点），快速衰减到 ~0.21
- **橙色实线**（`(1,0)`）：从 0 上升到 ~0.29 附近
- **绿色实线**（`(0,1)`）：从 0 上升到 ~0.21
- **红色实线**（`(1,1)`）：从 0 上升到 ~0.29

100 步以内就有大致形状，500 步以后基本贴合一个稳定值。把终端打印的 `d_π` 数值和实线最后停留的位置对照，就能直观看到**经验占比收敛到 stationary distribution**。

### 故障诊断

| 现象 | 可能原因 |
|------|---------|
| 跑出 `AssertionError: Each row of PI must sum to 1` | `PI` 改坏了（哪一行加起来不是 1）|
| 4 条曲线全收敛到 0.25 | 策略设成均匀了——可以但无聊，改成非均匀 |
| 一条曲线一直停在 1.0（其他全是 0）| 起点那个状态是"吸收态"——所有动作都 stay 或撞墙 |
| 曲线震荡不收敛 | 链是**周期**的（比如 deterministic 左右摇摆），没有收敛到唯一稳态 |
| 曲线还在动，没收敛 | 步数不够；把 `NUM_STEPS` 调到 5000 或 10000 |

---

## 七、和笔记的对应

笔记 line 4141–4142 的核心论述：
> *"Since more frequently visited states have higher values of d_π(s), their weights in the objective function are also higher than those rarely visited states."*

→ 跑完本示例后你应该直观感受到：**"哪些状态被经常走到"是由策略本身决定的**。把策略从均匀改成偏向某一侧，理论 `d_π`（终端打印）就偏向那一侧——agent 真的会把更多时间花在那里。这就是为什么 value function approximation 的目标函数要用 `d_π` 加权、而**不是**均匀加权（笔记 line 4126 提到的 drawback）。

---

## 八、扩展

- 把 `NUM_STEPS` 调到 10000，看曲线怎么更贴合虚线
- 把 `START_STATE` 改成 1 / 2 / 3，看**起点变了曲线还是收敛到同一个 `d_π`**（这是 stationarity 的关键性质——链不可约的话起点不影响长程分布）
- 写一个把 `PI` 改成完全均匀的版本，确认 `d_π = (0.25, 0.25, 0.25, 0.25)`
- 把网格改成 3×3，自己改 `STATES` 和 `build_transition_table`——同样思路

---

## 九、文件清单

| 文件 | 说明 |
|------|------|
| `train.py` | 主脚本：策略、转移表、仿真、画图都在这一个文件里 |
| `pyproject.toml` | Ruff 配置 |
| `requirements.txt` | numpy + matplotlib |
| `.vscode/settings.json` | VS Code Ruff 配置 |
| `README.md` | 本文件 |
