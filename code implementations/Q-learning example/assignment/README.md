# Q-learning 编程作业

> 对应笔记：`RL_note/RL_note_WestLakeU/Sean's notes/RL_WLU_note.typ` §**"5. TD Learning of Optimal Action Values : Q-learning"**（约第 3691 行）。

---

## 一、题目背景

Sarsa 估计**给定策略**的 Q 值（on-policy）；Q-learning **直接估计最优** Q 值（off-policy）。两者公式只差**一个 max**：

```
Sarsa  :  TD target = r + γ · Q[s_next, a_next]   ← 行为策略实际走的动作
Q-learn:  TD target = r + γ · max_a Q[s_next, a]  ← 所有动作里最大的那个
```

**唯一这一行替换**带来两个核心变化：
1. **解耦**：TD target 不再依赖行为策略实际走的动作——π_b 可以是**任意**策略
2. **训练表现 ≠ 学到的策略**：off-policy 让"训练时的回合长度/奖励"和"最终箭头矩阵"完全不同（详见 §六）

**本次作业**：复用 Sarsa 那张 5×5 网格，实现 Q-learning，跑 4 组行为策略实验（基线 + ε 两档 + 偏置）。

---

## 二、环境与参数

**故意与 Sarsa example 同款**——这样才能在完全相同的条件下做算法对比。

| 参数 | 值 | 说明 |
|------|----|------|
| 网格 / 起点 / 终点 / 禁区 / 奖励 / max_steps / α / γ / episodes | **全部同 Sarsa**（故意一致） | 便于在同一环境下做算法对比 |
| **行为策略 π_b** | **`uniform` / `epsilon_greedy` / `biased_left`** 三选一 | **Q-learning 特有的可选项**——Sarsa 只能用 ε-greedy |

**所有可调常量都标了 `# 可调`**，集中在 `assignment/train.py` 和 `assignment/agent.py` 顶部。

---

## 三、你要写的内容（全部在 `agent.py`）

只有**1 处 TODO**——Q-learning 的核心公式：

| 位置 | 方法名 | 写什么 | 难度 |
|------|--------|-------|------|
| TODO 1 | `learn(s, a, r, s_next, a_next)` | **TD target 第二项**（把那一行里的 `None` 换掉）| ⭐⭐ |

**`select_action` 和 `run_episode` 都直接给好**——和 Sarsa 那两个几乎一样，区别只在 `learn` 里那一行。

> ℹ️ `learn` 签名里的 `a_next` 参数**保留但不用**——Q-learning 不依赖"行为策略实际走的动作"。**保留是为了和 `SarsaAgent.learn` 签名完全一致**，方便你对照读两个 agent.py 找出**唯一差异**。

---

## 四、必做的调参实验 ⭐必做

**4 组实验**（基线 uniform + ε 两档 + biased_left）——每条都体现 Q-learning 特有的现象。**对照 Sarsa 是可选的扩展**，见实验 4。

### 实验 1（基线）：uniform π_b

```python
# train.py 顶部
from agent import QLearningAgent

agent = QLearningAgent(num_states=env.num_states(), behavior_policy="uniform", ...)
```

**期望观察（实测，seed=0/1/42）**：10_000 步的 episode 走遍全部 19 个可进入格子；state value error 从 103.2 降到 ≈ 60–70（**值还没收敛**），但 **19 个箭头已经全部是最优动作**，target 格子是圆点（stay）。**策略先于值收敛**；改成 100_000 步，error 就降到 ≈ 0。

**为什么**：uniform π_b **训练时永远随机走**，从不走近 target；但 Q-learning 的 **max 让 Q 表直接逼近最优**——所以**离线学到的策略**和**训练时表现**完全不同。这是 off-policy 的**最直观体现**：行为策略差 ≠ 学习结果差。

### 实验 2：ε-greedy π_b —— 比较 ε=0.1 和 ε=0.3

```python
# train.py 顶部，跑两次：EPSILON=0.1 和 EPSILON=0.3
EPSILON = 0.1   # 先跑这个
EPSILON = 0.3   # 再跑这个
```

**期望观察（实测，seed=0/1/42）**：

| ε | 走到的格子数 | state value error（起点 → 末值）| 19 个箭头里最优的个数 | Q_max |
|---|------------|------------------------------|--------------------|-------|
| **0.1** | 6 | 103.2 → 103.2（**完全不动**）| 8 | **0** |
| **0.3** | 6 | 103.2 → 103.2（**完全不动**）| 8 | **0** |
| 0.5 | 17–19 | 103.2 → ≈ 67 | 13–15 | 10 |

ε=0.1 和 0.3 时 **Q 表里连一个正值都没有**（`Q_max = 0`），说明 agent 从头到尾没拿到过 target 的 +1 奖励。

**为什么会这样（这是本实验的核心）**：

1. Q 表初始全 0 → `argmax` 恒返回动作 0（"下"）→ agent 开局就**一路向下**，钻进左列 + (1,2) 这 6 个格子组成的"口袋"（出口只有 (0,0) 往右一条）
2. 贪心部分一直把它往口袋里拉，随机部分偶尔把它带到口袋口，**不足以把它送出去**
3. **关键**：既然从没到过 target，就永远拿不到正奖励 → Q 表永远学不到"该往右上走" → 策略不变 → 继续困住

这是一个**自我强化的死锁**，**只跟探索强度有关**。ε 要提到 0.5 才走得出去，而且 10_000 步里策略也只学对一部分。uniform（相当于 ε=1）才是最好的数据来源。

**这正是笔记 line 3888 的结论**："探索性越强，探索越充分，最后才能很好地收敛。"

### 实验 3：biased_left π_b

```python
agent = QLearningAgent(..., behavior_policy="biased_left")
```

**期望观察（实测）**：和 ε=0.1 同样**完全失败**：state value error 103.2 不动，`Q_max = 0`。图 1 的热力图只有左侧 9–11 个格子有颜色（70% 概率往左），右侧大片格子访问次数为 0。

**为什么**：`biased_left` 把 70% 的概率压在"左"上，实际可用的探索只有 30% 且分散——比 ε=0.1 的 ε-greedy 还要差。

### 四种 π_b 汇总对比表（实测，10_000 步默认参数）

| π_b | 跑过的格子数 | State value error 末值 | 19 个箭头是否成体系 | 备注 |
|-----|------------|---------------------|------------------|------|
| `uniform` | 19 / 19 | ≈ 60–70（值未收敛）| ✅ 19/19 最优 | 50_000 步 → ≈ 1；100_000 步 → ≈ 0 |
| `epsilon_greedy` ε=0.1 | 6 / 19 | 103.2 不动 | ❌ 8/19 | agent **死锁**，Q 表全 0 |
| `epsilon_greedy` ε=0.3 | 6 / 19 | 103.2 不动 | ❌ 8/19 | 仍然死锁 |
| `biased_left` | 9–11 / 19 | 103.2 不动 | ❌ 10/19 | 70% 往左，右侧从未访问 |

（25 个格子里 6 个 forbidden 进不去，所以可进入的是 19 个。）

**两个要点**：

1. **state value error 末值才是真指标**。Q-learning 不再有"训练曲线"——我们删掉了 Episode length / Total rewards 这两张 Sarsa 风格的图。取而代之的是 **state value error**，它直接度量 **Q 和 ground truth V\* 的差距**——这才是"学得怎么样"的客观指标。
2. **探索强度决定成败**。同一算法、同一环境、同样 10_000 步，uniform 学出完整最优策略，ε=0.1/0.3 一步都没学到。

### 实验 4（可选）：4 组实验的 state value error 叠图

把 4 组的 **state value error** 曲线**叠在同一张图**上——最直观地看出"探索强度"的影响：

```python
# 在 train.py 里跑 4 次，每次换参数，把 errors 存下来
configs = [("uniform", 0.1), ("epsilon_greedy", 0.1),
           ("epsilon_greedy", 0.3), ("biased_left", 0.1)]
# 然后画 4 条 state value error 曲线对比
```

**这张图能看出**（实测，10_000 步默认参数，3 seed 平均）：

| 配置 | error 起点 | error 中段（5000 步）| error 末值 | 结论 |
|------|---------|-------------------|----------|------|
| uniform | 103.2 | ≈ 84 | ≈ 63 | 稳步下降（值未收敛，策略已最优）|
| ε=0.1 | 103.2 | 103.2 | 103.2 | **完全不动**，Q 表死锁 |
| ε=0.3 | 103.2 | 103.2 | 103.2 | **完全不动**，仍然死锁 |
| biased_left | 103.2 | 103.2 | 103.2 | **完全不动**，右侧格子从未访问 |

**关键观察**：4 条曲线里只有 **uniform** 在下降，其余 3 条是一条水平线。error 不动说明 Q 表压根没接触到 target 的奖励，**直接证明 Q 表学没学对**，而不只是"agent 走得快不快"。

→ **结论**：state value error 才是 Q-learning 学习的客观指标——比 Sarsa 的训练曲线更能说明"Q 表学得怎么样"。

### 对比记录表（提交时填好）

```
==================================================================
                  行为策略           | 总步数    | 观察（一句话）
------------------+------------------+-----------+---------------------
基线 (uniform)     | uniform π_b      | 10_000    |
------------------+------------------+-----------+---------------------
Exp 2a (ε=0.1)     | ε-greedy         | 10_000    |
------------------+------------------+-----------+---------------------
Exp 2b (ε=0.3)     | ε-greedy         | 10_000    |
------------------+------------------+-----------+---------------------
Exp 3 (biased_left)| "左" 0.7         | 10_000    |
==================================================================

每组要填：state value error 末值 / 箭头矩阵是否成体系 / 热力图覆盖了哪些格子（一句话）

是否符合"期望观察"？
基线: 是 / 否，原因: ___
Exp 2a: 是 / 否，原因: ___
Exp 2b: 是 / 否，原因: ___
Exp 3: 是 / 否，原因: ___

关键思考（必填）：
(1) 为什么 ε=0.1 和 ε=0.3 完全学不会、uniform 就学会了？（提示：看图 1 热力图覆盖了哪些格子）
(2) uniform π_b 训练时表现比 ε-greedy 差，但学到的 Q 表反而好——
    为什么"训练曲线"（state value error 曲线）判断得了 Q-learning 学得怎么样？
```

---

## 五、操作步骤

```bash
# 1. 进入作业目录
cd "E:/RL-西湖大学赵世钰/RL-WestlakeU-Zhao-main/RL_note/RL_note_WestLakeU/code implementations/Q-learning example/assignment"

# 2. 装依赖（只需要 numpy + matplotlib）
pip install -r requirements.txt

# 3. 打开 agent.py，把 TODO 1 的占位符换掉

# 4. 跑基线
python train.py

# 5. 改 train.py 顶部参数再跑，依次做 Exp 2a / 2b / 3：
#    BEHAVIOR_POLICY="epsilon_greedy", EPSILON=0.1     → Exp 2a
#    BEHAVIOR_POLICY="epsilon_greedy", EPSILON=0.3     → Exp 2b
#    BEHAVIOR_POLICY="biased_left"                     → Exp 3
```

应该弹**三张图**。3 次实验 + 1 次基线 = 共 4 组图，建议截图保存。

---

## 六、期望看到什么

### 关键先理解：3 张图对齐 Zhao 课件

我们出的 3 张图**严格对齐** Zhao `Ch7_Qlearning_eg_uniform_pi_b.png` 的布局（见 `Sean's notes/images/`）：

| 我们的图 | Zhao PPT 的图 | 内容 |
|----------|---------------|------|
| 图 1 | (b) Generated episode | 5×5 网格，每个格子按**访问次数**填热力图颜色 |
| 图 2 | (b) Q-learning result | **State value error** 曲线（横轴 = step in the episode）|
| 图 3 | (a) Estimated policy | **全网格箭头矩阵**（**没有**蓝色贪心轨迹）|

### 图 1：Generated episode（访问次数热力图）

- π_b 从左上角起点 (0,0) 出发，生成**一条**长度 `NUM_TRAINING_STEPS = 10,000` 的 episode；到了 target **不结束**，继续走
- agent 每一步从一个格子的**中心**走到相邻格子的**中心**；图中统计每个格子被占据的步数（包括起点 s₀，撞墙/撞禁原地不动也算一次）
- 颜色**填满整个格子**，越深表示访问越多；格子中心的数字是次数，右侧 colorbar 给刻度。所有格子的次数加起来 = 10,000
- forbidden 格子保持橙色（我们的 env 里撞禁会被弹回，进不去，次数恒为 0）；target 用**蓝色粗边框**标出

**为什么这么画**：这张图对应 Zhao 的"generated experience"：π_b 生成的这条 episode **就是 Q-learning 的全部训练数据**。热力图直接告诉你哪些格子被充分采样、哪些根本没采到，也就是**探索质量**。

- **基线 uniform**：19 个可进入格子**全部有颜色**，每格 400–650 次左右，分布比较均匀
- **ε=0.1 / 0.3 失败**：只有左列 + (1,2) 这 6 个格子有颜色，其余全是 0（agent 死锁的证据）
- **biased_left 失败**：颜色**集中在左侧**，右侧大片格子是 0

### 图 2：State value error（vs step）

- 横轴 = **step in the episode**（不是 episode index），和 Zhao 一致
- 纵轴 = `Σ_s |max_a Q(s, a) − V*(s)|`，对 19 个可进入格子求和。V\* 是 **ground truth**（用 Bellman optimality 通过 value iteration 直接解出，**与 Q-learning 本身独立**）；因为到了 target 不结束、待在 target 每步 +1，V\*(target) = 1/(1−γ) = 10
- 起点 = Σ V\* ≈ 103.2（Q 初始全 0），**步数足够时降到 0**

**不同 π_b 的差异**：
- **uniform**：误差稳步下降，10,000 步后 ≈ 60（**值未收敛**，见下框）
- **ε=0.1 / 0.3 失败**：一条水平线，Q 永远学不到 target 附近的值
- **biased_left 失败**：同样是水平线

> ⚠️ **关于"10,000 步末值 ≈ 60"**：Q 从 0 逐步涨到 V\*（最大 10），α=0.1、γ=0.9 时要很多步才能涨满。这是**步数限制**，不是 bug：`NUM_TRAINING_STEPS = 50_000` 末值 ≈ 1，`100_000`（Zhao PPT 的规模）末值 ≈ 0。但**策略在 10,000 步时已经全对**（见图 3），这正是"策略先于值收敛"。

### 图 3：Estimated policy（**全网格箭头矩阵**，**没有蓝色轨迹**）

- **每个非 forbidden 格子画 argmax Q 的动作**，共 19 个（含 target）
- 移动动作画箭头，stay 画圆点：target 学出来的是 **stay**（原地每步 +1）
- **没有从 (0,0) 出发的蓝色贪心轨迹**，这是 **Q-learning 与 Sarsa 的关键视觉差异**！

**为什么没有蓝色贪心轨迹**：Q-learning 学到的是 **Q\*，对所有格子都成立的最优动作**，不是"从某起点走一次贪心"的策略。Q\* 的视觉化就是**完整的箭头矩阵**——画一条从 `(0,0)` 出发的轨迹反而会**误导**（让人以为 Q-learning 只对那个起点有意义）。

**对比 Sarsa 镜像图**：Sarsa 那张画了从 `(0,0)` 出发的蓝色轨迹——这是**正确的**因为 Sarsa 学的是 Q^π，π 只能从 start 出发才有意义。Q-learning 的图**反过来不画轨迹**才能体现 off-policy 的特点。

**不同 π_b 的差异**：
- **uniform**：19 个动作**全部最优**，例如 `(0,1)` 向上、`(3,2)` 向右、`(4,4)` 向左，target 是圆点
- **ε=0.1 / 0.3 失败**：Q 几乎全 0，argmax 默认动作 0（"下"），**箭头大多朝下**，19 个里只有 8 个碰巧对
- **biased_left 失败**：和 ε=0.1 类似，右侧没访问过的格子全是默认的"下"

### 故障诊断

| 现象 | 可能原因 |
|------|---------|
| **跑出 `TypeError: float * NoneType`** | TODO 1 没填对（确认 `np.max(self.Q[s_next])` 已填）|
| **图 3 出现一条从 (0,0) 出发的蓝色轨迹** | 这是 Sarsa 风格的残留——Q-learning **不应该**画这条轨迹 |
| **图 1 forbidden 格子没有访问次数** | 正常，我们的 env 撞禁会原地弹回，forbidden 进不去 |
| **图 2 state value error 一直是 103.2 不动** | 探索不足（Exp 2 / Exp 3 现象）；如果 uniform 也不动，就是 TODO 1 填错了 |
| **图 3 19 个箭头大多是 "下"** | Q 表大片 0 → argmax 默认 0，**探索不足**（Exp 2 / Exp 3 现象）|
| **图 2 末值 ≈ 60 没到 0** | **正常**：默认 10,000 步不够，改成 100,000 即可 |

---

## 七、Sarsa vs Q-learning 对照表

**这张表是 Q-learning 学习的核心资源**——做完 TODO 后对照读一遍。

| 项 | Sarsa | Q-learning |
|---|---|---|
| **TD target 第二项** | `Q[s_next, a_next]` | `max_a Q[s_next, a]` |
| **动作来源** | 行为策略实际走 | 所有动作中最大 |
| **是否用 `a_next`** | 用 | 不用（参数保留但不用）|
| **行为策略 π_b** | 必须和目标策略对应（ε-greedy）| **任意**（uniform / ε-greedy / 偏置都行）|
| **on/off-policy** | on-policy | **off-policy** |
| **数学问题** | 给定 π 解 Bellman equation | 解 **Bellman optimality equation** |
| **收敛到** | 当前策略的 Q 值 | **最优 Q 值**（如果探索充分）|
| **TD target 是否相关** | 行为策略有关 | **无关**——max 解耦了 |
| **"训练曲线"vs"Q 表"**| 两者同步（on-policy）| **解耦**——训练曲线可能很糟但 Q 表学得很好（off-policy 特性）|

**核心洞察**：Q-learning 用 max 让 TD target **完全脱离行为策略**——这意味着
- 行为策略可以**随便设计**（甚至完全偏离最优策略）
- 但**行为策略的覆盖范围仍决定 Q 表能不能学到东西**（笔记 line 3888，Exp 3 验证）

---

## 八、提交物建议

**必交**：

1. **`agent.py`**：填完的 TODO 1 代码
2. **基线的三张图**（generated episode 热力图 + state value error 曲线 + 箭头矩阵）
3. **§四 的对比记录表**——4 组实验的观察和判断

**可选（做完能加分）**：

4. **§七 对照表**——你能用自己的话解释清楚每一行吗？
5. **§四实验 4** 的叠图——4 组配置的 **state value error 曲线**对比

---

## 九、扩展（学有余力时做）

- **对照 Sarsa**：把 `RL_note/RL_note_WestLakeU/code implementations/Sarsa example/solution/agent.py` 的 `SarsaAgent` 拿到本目录，用**同样的 seed / ε=0.1** 跑一次——实测 **Sarsa 442/500 命中 target，而 Q-learning 0/500**。同一环境、同一探索强度，只差 TD target 那一项，结果却截然相反。（想清楚为什么，比做完 4 组实验更有收获）
- 把 γ 调到 0.5 + uniform π_b——看 Q-learning 对远见的敏感度
- 实现 Q-learning 的 off-policy 形式（笔记 line 3850–3858）：行为策略采样，目标策略永远 greedy——这是 Q-learning 最纯粹的 off-policy
- 把 `EPSILON` 从 0.1 逐步调到 0.3，找出"刚好能学会"的临界值——粗测在 0.2 和 0.3 之间

---

## 十、参考

- 笔记原文：`RL_note/RL_note_WestLakeU/Sean's notes/RL_WLU_note.typ` §5（约 3691–3891 行）
- Sarsa 作业：`RL_note/RL_note_WestLakeU/code implementations/Sarsa example/`（同一网格、同一 train.py——直接对照）
- Zhao 课件：`Lecture slides/slidesForMyLectureVideos/L7-Temporal-difference learning.pdf`