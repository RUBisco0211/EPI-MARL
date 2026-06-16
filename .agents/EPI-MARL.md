# EPI-MARL 仓库技术分析

## 1. 仓库定位

本仓库实现论文 **Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form** 中的部分核心思想。当前真正接入主训练入口的内容是：

- 连续时间 Multi-Particle Environment（MPE）；
- EPI 的集中训练、分散执行（CTDE）网络结构；
- return critic、constraint critic、动力学模型、代价模型；
- epigraph HJB residual、rollout target、value-gradient iteration（VGI）；
- 基于 Hamiltonian 最小化的 actor 更新。

仓库还包含 Safe Multi-Agent MuJoCo、耦合振子 didactic 环境、Replay Memory、状态构造工具等代码，但它们大多没有接入当前的 `main.py` 训练流程。

因此，应将仓库理解为：

> 一个以 EPI + 连续时间 MPE 为主的研究代码快照，而不是论文全部实验的一键复现框架。

---

## 2. 目录与模块职责

```text
Epi-CT-MARL/
├── main.py                         # 顶层训练入口，与 algo/main.py 基本相同
├── algo/
│   ├── main.py                     # EPI + MPE 主训练入口
│   ├── epigraph_pinn/
│   │   ├── epi_pinn_agent.py       # EPI agent、各类损失与更新流程
│   │   └── network.py              # actor、critics、动力学/代价网络
│   ├── memory.py                   # 多种 replay buffer；当前主流程未使用
│   ├── utils.py                    # 网络更新、目标状态构造、return 计算等
│   ├── normalized_env.py           # Gym action/observation wrapper；主流程未使用
│   └── random_process.py           # OU noise；主流程未使用
├── continuous_env/
│   ├── make_env.py                 # 当前主流程的 MPE 环境工厂
│   ├── multiagent/
│   │   ├── core.py                 # MPE 世界、实体、碰撞与连续时间积分
│   │   ├── environment.py          # Gym 风格多智能体环境与 step_con
│   │   ├── scenarios/              # MPE 场景定义
│   │   └── rendering.py            # pyglet 渲染
│   ├── multiagent_mujoco/          # MAMuJoCo 原始/兼容实现
│   ├── Safe_Mujoco/                # 带安全墙体和连续时间接口的 MuJoCo 变体
│   └── didactic/                   # LQR/CBF/耦合振子测试环境
├── requirements.txt
└── README.md
```

### 当前实际调用链

```text
main.py / algo/main.py
  -> continuous_env.make_env.make_env()
    -> scenarios/<scenario>.Scenario.make_world()
    -> MultiAgentEnv
  -> epi_agent_new
    -> PolicyNet / ValueNet / DynamicsNet / RewardNet
  -> episode rollout with env.step_con(action, dt)
  -> model.update(full_episode_batch)
  -> model.save_model()
```

`main.py` 和 `algo/main.py` 当前内容基本重复。README 示例使用顶层 `main.py`。

---

## 3. 算法总体设计

### 3.1 问题表达

算法把安全连续时间控制问题拆成两个主要量：

- `V_ret(x)`：从状态 `x` 出发的累计任务代价；
- `V_cons(x)`：从状态 `x` 出发的未来最坏约束违反程度。

代码使用 epigraph 辅助值：

```text
V_tilde(x, z*) = max(V_cons(x), V_ret(x) - z*)
```

其中：

- return 分支激活时，策略主要优化任务表现；
- constraint 分支激活时，策略主要减少约束违反；
- actor 根据 `V_tilde` 对状态的梯度和学习到的动力学/代价模型构造 Hamiltonian。

### 3.2 CTDE 结构

代码采用集中训练、分散执行：

- 每个 agent 有独立 actor；
- actor 输入本地观测 `o_i` 和当前时间间隔 `dt`；
- critics 和模型输入所有 agent 观测拼接成的集中状态以及联合动作；
- 每个 agent 有独立的 return critic、constraint critic、动力学模型和代价模型。

设：

- `N`：agent 数量；
- `D`：单 agent 观测维度；
- `A`：单 agent 动作维度；
- `B`：轨迹长度，即当前实现中的 batch size。

主要张量形状：

| 数据 | 形状 |
|---|---|
| rollout state | `[B, N, D]` |
| centralized state | `[B, N * D]` |
| joint action | `[B, N * A]` |
| dt | `[B, 1]` |
| per-agent reward/cost | `[B, N]` |
| per-agent constraint signal | `[B, N]` |

---

## 4. 主训练流程

主流程位于 `main.py` 和 `algo/main.py`。

### 4.1 初始化

1. 使用 `make_env(args.scenario, args.seed)` 创建 MPE。
2. 从环境读取：
   - `n_agents = env.n`
   - `n_actions = env.world.dim_p`
   - `n_states = env.observation_space[0].shape[0]`
3. 创建 `epi_agent_new`。
4. 设置随机种子和 PyTorch deterministic 标志。
5. 设置总物理时长 `T_total = 5.0`。

### 4.2 不规则时间间隔

每个 episode 使用 Dirichlet 分布重新采样时间步：

```python
delta_ts = rng.dirichlet(np.ones(num_steps)) * T_total
```

因此：

- 每个 `dt > 0`；
- 一个 episode 内所有 `dt` 之和固定为 `T_total`；
- actor 显式接收当前 `dt`，以学习对时间分辨率敏感的策略。

代码还计算了 `time_to_go`，但当前训练流程没有使用它。

### 4.3 Rollout

每一步执行：

```text
action = model.choose_action(state, dt)
next_state, [continuous_reward, discrete_reward], done, constraints
    = env.step_con(action, dt)
```

训练轨迹保存：

- 当前和下一状态；
- 联合动作；
- 连续任务 reward；
- `dt`；
- 环境返回的连续约束信号；
- rollout 中传播的辅助变量 `z`。

需要注意：

- 日志中的 `total_reward` 使用 `continuous_reward + discrete_reward`；
- critic 学习时只将 `continuous_reward / normal_factor` 放入 `trajectory_rewards`；
- 离散碰撞惩罚没有直接进入 return critic 的训练目标；
- 连续约束信号通过 `trajectory_constraints` 单独训练约束网络。

### 4.4 Return 计算

`compute_discounted_returns()` 从轨迹末尾反向计算：

```text
G_t = reward_t * dt_t + gamma^dt_t * G_(t+1)
```

这使折扣和运行代价都随实际物理时间间隔变化。

### 4.5 Episode 后更新

当前实现不是典型 replay-buffer/off-policy 更新。每个 episode 结束后，整条轨迹直接作为 batch：

```text
full episode rollout
  -> model.update(batch)
```

更新顺序固定为：

1. `dynamics_training`
2. `reward_training`
3. `const_training`，重复 10 次
4. `value_training_epigraph`
5. `single_cost_training`
6. `tilde_value_training`
7. `target_vgi_training`
8. `policy_training`

虽然 `epi_agent_new` 创建了 `continuous_ReplayMemory`，但当前主流程没有向其中写入或采样。

---

## 5. 网络设计

网络定义位于 `algo/epigraph_pinn/network.py`。

### 5.1 AdaptiveTanh

`AdaptiveTanh` 实现：

```text
tanh(a * x)
```

其中 `a` 是可学习参数，可以按层共享或按神经元独立。它被用于模型网络和 actor 的部分层，使激活函数斜率可适配训练数据。

### 5.2 PolicyNet

每个 agent 一个策略网络：

```text
input: local observation D + dt
FC(128) + ReLU
FC(128) + ReLU
FC(64)  + AdaptiveTanh
FC(A)   + AdaptiveTanh
```

训练时：

- 网络输出作为高斯分布均值；
- 使用固定协方差矩阵进行重参数化采样；
- 动作裁剪到 `[-1, 1]`。

eval 模式下，`choose_action()` 直接使用均值动作，不采样噪声。

### 5.3 ValueNet

`ValueNet` 用于两类 critic：

- `value_nets[i]`：return critic；
- `cost_nets[i]`：constraint critic。

结构：

```text
input: centralized state N * D
FC(128)
FC(128)
FC(1)
```

隐藏层根据 `args.relu` 使用 ReLU 或 Tanh。

### 5.4 DynamicsNet

每个 agent 一个动力学模型：

```text
input: centralized state + joint action + dt
output: predicted next centralized state
```

代码随后通过：

```text
f_hat = (predicted_next_state - current_state) / dt
```

得到连续时间动力学导数近似。

### 5.5 RewardNet

`RewardNet` 被复用于两种模型：

- `reward_nets[i]`：预测任务运行代价；
- `single_cost_net[i]`：预测即时连续约束信号。

输入为：

```text
centralized state + joint action + dt
```

输出为标量。

---

## 6. EPI Agent 的各训练组件

核心代码位于 `algo/epigraph_pinn/epi_pinn_agent.py`。

### 6.1 动力学模型训练

`dynamics_training()` 使用 MSE：

```text
DynamicsNet(x_t, u_t, dt_t) ~= x_(t+1)
```

每个 agent 有一个独立模型，但所有模型都预测完整 centralized next state。

### 6.2 任务代价模型训练

`reward_training()` 将环境连续 reward 取负，作为任务代价：

```text
RewardNet_i(x_t, u_t, dt_t) ~= -reward_i
```

### 6.3 即时约束模型训练

`single_cost_training()` 拟合环境返回的连续约束信号：

```text
single_cost_net_i(x_t, u_t, dt_t) ~= constraint_i(t)
```

该模型主要用于 VGI 和 actor 的 constraint 分支。

### 6.4 Constraint critic

`const_training()` 首先对每个时间点计算未来后缀最大约束：

```text
c_max(t) = max_{tau >= t} c(tau)
```

然后训练：

```text
V_cons_i(x_t) ~= c_max_i(t)
```

这对应“未来最坏约束违反程度”的语义。

当前每个 episode 会连续执行该训练 10 次。

### 6.5 Return critic

`value_training_epigraph()` 的注释称其为 TD(0)，但实际实现直接拟合整条轨迹计算出的 discounted return：

```text
V_ret_i(x_t) ~= -discounted_return_i(t)
```

因此它当前更接近 Monte Carlo return regression，而非带 bootstrap 的 TD(0)。

### 6.6 `z*` 与分支选择

`compute_z_star()` 当前使用启发式公式：

```text
z* = V_ret + relu(V_cons) + z_bias
```

随后构造：

```text
V_tilde = max(V_cons, V_ret - z*)
```

代码根据两项大小创建 branch mask：

- `V_ret - z* > V_cons`：return 分支；
- 否则：constraint 分支。

注意：

- 函数注释和实际公式存在不一致；
- `z*` 的范围裁剪代码被注释；
- 当前实现不是论文附录描述的一维搜索。

### 6.7 Epigraph HJB residual

`tilde_value_training()`：

1. 构造 `V_tilde`；
2. 用 autograd 计算 `grad_x V_tilde`；
3. 用真实 rollout 差分近似：

   ```text
   f(x, u) ~= (x_(t+1) - x_t) / dt
   ```

4. 根据激活分支设置：

   ```text
   partial_z V_tilde = -1  # return branch
   partial_z V_tilde = 0   # constraint branch
   ```

5. 构造 residual：

   ```text
   max(
       future_max_constraint - V_tilde,
       grad_x(V_tilde) dot f - partial_z(V_tilde) * task_cost
           + log(gamma) * V_tilde
   )
   ```

6. 最小化 residual 的平方。

该步骤同时更新 return critic 和 constraint critic。

### 6.8 Value Gradient Iteration

`target_vgi_training()` 直接约束 value gradient，而不仅约束 value：

1. 对当前 `V_tilde` 求 `grad_x V_tilde(x_t)`；
2. 从任务代价模型和约束模型得到代价梯度；
3. 根据 epigraph 激活分支选择任务梯度或约束梯度；
4. 对 learned dynamics 求 Jacobian-vector product；
5. 构造下一步传播得到的目标梯度；
6. 最小化当前 value gradient 与目标梯度之间的 MSE。

这部分是当前实现中计算最复杂、二阶自动微分依赖最强的部分。

### 6.9 Actor 更新

`policy_training()` 为每个 agent 构造 epigraph Hamiltonian：

```text
H =
    grad_x(V_tilde) dot f_hat(x, u)
    - partial_z(V_tilde) * learned_task_cost(x, u)
    + learned_constraint_cost(x, u) * constraint_branch_mask
    + log(gamma) * V_tilde
```

actor 通过可微的动作采样最小化：

```text
actor_loss = mean(H * dt)
```

其中：

- actor 只接收本地 observation + `dt`；
- Hamiltonian 使用 centralized state 和 joint action；
- 当前逐 agent 更新时，会把该 agent 新采样动作写入 rollout joint action，其余 agent 动作保持轨迹动作。

---

## 7. MPE 连续时间改造

### 7.1 原始 MPE 结构

`continuous_env/multiagent/core.py` 保留了经典 MPE 的基本对象：

- `Entity`：实体通用属性；
- `Agent`：可移动、可通信实体；
- `Landmark`：目标或障碍物；
- `World`：实体集合、物理参数和状态推进；
- soft contact force：处理实体之间的碰撞响应。

### 7.2 连续时间接口

原始固定步长路径：

```text
MultiAgentEnv.step()
  -> World.step()
  -> integrate_state(..., world.dt=0.1)
```

新增连续时间路径：

```text
MultiAgentEnv.step_con(action_n, dt)
  -> World.step_con(dt)
  -> integrate_state_continuous(..., dt)
```

连续积分执行：

```text
velocity <- velocity * (1 - damping)
velocity <- velocity + force / mass * dt
position <- position + velocity * dt
```

额外行为：

- 对速度执行 `max_speed` 裁剪；
- 将位置裁剪到 `[-1, 1]`；
- 检测 NaN 位置并重置为零。

### 7.3 `step_con()` 的返回协议

场景的 `reward()` 需要返回三元组：

```text
(task_reward, discrete_collision_reward, continuous_constraint_signal)
```

`MultiAgentEnv.step_con()` 将其整理为：

```text
next_obs,
[per_agent_task_reward, per_agent_discrete_reward],
per_agent_done,
per_agent_continuous_constraint
```

这种拆分使：

- 任务表现和离散碰撞可用于日志；
- 连续约束信号可单独训练 constraint critic；
- epigraph 不必直接对离散碰撞指示函数求梯度。

### 7.4 动作处理

`MultiAgentEnv` 的 `action_space` 默认仍声明为离散空间，但 `_set_action()` 在非 `discrete_action_input` 路径下会直接把 actor 输出向量设置为二维物理动作。

随后动作乘以：

```text
sensitivity = agent.accel or 2.0
```

因此当前 EPI 实际输出连续二维控制，即使 Gym action-space 元数据仍沿用旧 MPE 的离散声明。

### 7.5 观测设计

各场景通常使用：

```text
self velocity
self absolute position
relative goal/landmark positions
relative obstacle positions
relative other-agent positions
optional communication or one-hot identity
```

actor 使用本地观测，critic 则把所有 agent 的观测直接 flatten 成 centralized state。这里的 centralized state 不是环境的最小真实状态，而是所有局部观测的拼接，因此可能包含重复信息。

---

## 8. MPE 场景设计

### 8.1 Formation

文件：`continuous_env/multiagent/scenarios/formation.py`

- 3 个 agent；
- 3 个固定目标，形成三角形；
- 2 个固定障碍；
- agent 初始位置和目标位置均固定；
- task reward：agent 到其指定目标的负欧氏距离；
- 离散安全惩罚：agent-agent 和 agent-obstacle 碰撞，每次 `-10`；
- 连续约束：对重叠深度使用正 hinge penalty；
- observation：自身速度/位置、自己的相对目标、障碍相对位置、其他 agent 相对位置。

这是 README 默认示例，也是当前实现最完整的场景之一。

### 8.2 Line

文件：`continuous_env/multiagent/scenarios/line.py`

- 3 个 agent；
- 3 个固定线形目标；
- 2 个障碍；
- task reward：到固定分配目标的负距离；
- 离散安全惩罚：agent-obstacle 与 agent-agent；
- 连续约束：对两类碰撞计算正 hinge penalty。

文件中保留了较多被注释的旧连续 penalty 实现。

### 8.3 Corridor

文件：`continuous_env/multiagent/scenarios/corridor.py`

- 3 个 agent；
- 3 个固定目标；
- 2 个较大圆形 landmark 作为 corridor walls；
- task reward：到指定目标的负距离；
- 离散惩罚当前只检查 agent-obstacle；
- agent-agent 离散碰撞代码被注释；
- 连续约束当前也是 obstacle-only。

连续约束在安全区仍返回负值：

```text
penetration <= 0 -> 0.5 * penetration
```

因此其语义是 signed safety margin，而不是非负 violation cost。

### 8.4 Target

文件：`continuous_env/multiagent/scenarios/target.py`

- 2 个 agent；
- 2 个固定目标；
- 1 个障碍；
- task reward：到指定目标的负距离；
- 离散和连续约束当前主要检查 agent-obstacle；
- observation 中包含 agent one-hot identity。

与论文描述相比，agent-agent 约束在当前实现中未启用。

### 8.5 Simple Spread / Cooperative Navigation

文件：`continuous_env/multiagent/scenarios/simple_spread.py`

- 3 个 agent、3 个 landmark；
- task reward 使用每个 landmark 到最近 agent 的平方距离；
- agent-agent 碰撞产生离散惩罚；
- 连续约束使用所有 agent pair 的 signed distance，并在碰撞时放大。

由于场景设为 collaborative，同一个全局覆盖任务 reward 会对每个 agent 重复返回。

### 8.6 Simple Tag / Cooperative Predator-Prey

文件：`continuous_env/multiagent/scenarios/simple_tag.py`

- 3 个可训练 predator；
- 一个 landmark 表示移动目标；
- task reward：predator 到目标的负距离；
- 约束：predator-predator 碰撞；
- 目标移动逻辑写在 `observation()` 内。

需要特别注意：环境每次为每个 agent 调用一次 `observation()`，因此一个环境 step 中目标可能移动多次；reset 构造 observation 时也会移动目标。这使目标运动依赖观测查询次数，而不是严格依赖物理时间步。

### 8.7 其他场景

`scenarios/` 还保留原始 MPE 的：

- `simple`
- `simple_adversary`
- `simple_crypto`
- `simple_push`
- `simple_reference`
- `simple_speaker_listener`
- `simple_world_comm`

这些场景多数仍使用原始 MPE reward 返回协议，不一定满足当前 `step_con()` 要求的三元组格式。

`single_navi.py` 是一个简单的单 agent 连续导航测试场景，返回协议兼容当前主流程。

---

## 9. MuJoCo 模块

仓库包含两套相关代码：

### 9.1 `continuous_env/multiagent_mujoco`

这是基于 MAMuJoCo 的多智能体关节划分实现，主要负责：

- 按 `agent_conf` 把机器人关节划分给不同 agent；
- 将 per-agent action 拼回 MuJoCo joint action；
- 构建局部观测和 global state；
- 提供 `step_con(actions, dt)` 的初步接口。

### 9.2 `continuous_env/Safe_Mujoco`

该目录包含安全 MuJoCo 环境变体：

- HalfCheetah；
- Ant；
- Hopper；
- Humanoid；
- ManyAgentAnt；
- ManyAgentSwimmer；
- CoupledHalfCheetah。

环境加入了墙体距离、碰撞 cost、连续安全 shaping，以及部分动态 `dt` 支持。例如 HalfCheetah 的 `step_con()` 会根据 `dt` 调整内部仿真帧数，并返回 wall collision cost。

### 9.3 当前状态

MuJoCo 模块没有接入 `main.py`：

- 当前环境工厂只创建 MPE；
- 主训练代码依赖 `env.world.dim_p`；
- MuJoCo `step_con()` 返回协议与 MPE 主流程不同；
- requirements 顶层文件未包含 MuJoCo 依赖；
- 代码依赖旧版 `gym` 和 `mujoco_py`。

因此 MuJoCo 代码目前更像独立环境实现和实验遗留代码，需要额外 adapter 才能用于 EPI 主流程。

---

## 10. Didactic 模块

`continuous_env/didactic/` 包含两类小型可解释环境：

### 10.1 TwoAgentLQRConstrained

- 简单双 agent 线性系统；
- 二次状态/动作代价；
- 顺序或间距约束；
- 使用 QP 求安全动作；
- 提供真实 value、value gradient 和单步 epigraph value。

### 10.2 CoupledOscillatorEnv

- 状态为 `[x1, v1, x2, v2]`；
- 两个 spring-damper agent；
- 二次任务代价；
- 约束为 `x1 - x2 - 0.02 <= 0`；
- 使用 CARE 得到 LQR；
- 使用 CBF half-space projection 得到安全 ground-truth controller；
- 提供 ground-truth value 和 gradient。

这些环境适合验证论文中的 value/action 对照，但当前没有接入 EPI 训练入口或自动绘图流程。

`didactic.py` 和 `didactic_2.py` 有较多重复实现，且部分 demo 调用签名与实际函数不一致，应视为研究草稿。

---

## 11. 辅助模块

### 11.1 `memory.py`

定义了多种环形 replay buffer：

- `ReplayMemory`
- `Uncertain_ReplayMemory`
- `continuous_ReplayMemory`
- `safe_ReplayMemory`
- `Epi_ReplayMemory`

当前主训练不使用这些 buffer。

### 11.2 `utils.py`

包含：

- tensor 转换；
- hard/soft target update；
- noisy action sampling；
- 为 spread、line、corridor、formation 构造理想目标状态；
- 连续时间 discounted return。

大多数目标状态构造函数没有被当前 EPI 主流程调用，且部分默认 agent/landmark 数量与当前场景定义不同。

### 11.3 `normalized_env.py`

包含 action normalization 和 observation 截断 wrapper，来自旧 DDPG/MPE 代码。当前主流程没有使用。

### 11.4 `random_process.py`

实现 Ornstein-Uhlenbeck exploration noise。当前 actor 使用 PyTorch `MultivariateNormal`，没有调用 OU process。

---

## 12. 模型保存与加载

训练保存目录：

```text
trained_model/<scenario>/<algo>/
```

保存内容：

- 每个 agent 的 policy；
- 所有 return value networks；
- 所有 dynamics networks；
- 所有 reward networks。

当前没有保存：

- constraint value networks；
- `single_cost_net`；
- optimizers；
- scheduler/episode 状态；
- RNG 状态。

此外，`load_model()` 使用固定的旧路径和旧文件命名，与 `save_model()` 不匹配。因此当前 checkpoint 适合提取 actor 做简单 rollout，但不能完整恢复 EPI 训练或约束分析。

---

## 13. 当前实现中的重要边界与风险

### 13.1 Eval 尚未实现

CLI 声明了 `--mode eval`，actor 也支持 eval 时使用确定性均值动作，但主循环遇到非 train 模式会直接抛出 `NotImplementedError`。

### 13.2 消融参数没有实际作用

CLI 定义了：

- `ablation_hjb`
- `ablation_target`
- `ablation_vgi`

但 `update()` 不读取这些参数，始终执行全部训练步骤。

### 13.3 探索噪声退火没有生效

`_decay_sigma()` 更新 `_sigma_scale`，但动作分布始终使用固定的 `self.cov_matrix`。因此 `_sigma_scale` 当前不会改变探索方差。

### 13.4 Target network 未实际使用

代码创建 `target_value_nets` 和 `soft_update_target_value_net()`，但：

- `target_value_nets` 是 Python list；
- soft update 函数按单一 module 调用 `.parameters()`；
- 主训练也没有调用该更新函数。

### 13.5 参数与命名存在遗留不一致

- `value_training_epigraph()` 注释描述 TD(0)，实际为 return regression；
- `compute_z_star()` 注释、公式和论文描述不一致；
- `time_to_go` 被计算但未使用；
- `ent_coef` 和 entropy 被构造但未加入 actor loss；
- `model_dir` 在主入口设置，但真正保存函数自行构造另一路径；
- `main.py` 与 `algo/main.py` 重复。

### 13.6 约束信号跨场景不统一

不同场景的连续约束可能是：

- 非负碰撞 hinge；
- 安全时为负、违反时为正的 signed margin；
- 所有 agent pair 共用的全局值；
- 每个 agent 独立的局部值。

constraint critic 将它们统一解释为“未来后缀最大值”，但跨场景数值尺度和零点语义并不完全一致。

### 13.7 环境随机性有限

多个 MPE 场景使用固定初始位置和固定目标位置。虽然 CLI 接收 seed，但 seed 对这些场景的实际影响有限。训练结果可能更接近单一初始条件下的轨迹优化。

---

## 14. 当前可运行实验的真实范围

按现有入口，最可靠的工作流是：

```bash
python main.py --algo epi --scenario formation --seed 113
```

可替换为当前三元 reward 协议兼容的 MPE 场景，例如：

```text
formation
line
corridor
target
simple_spread
simple_tag
single_navi
```

当前主入口能够完成：

- 不规则 `dt` 的 MPE rollout；
- EPI 网络训练；
- episode total reward 打印；
- 周期性保存部分网络。

当前主入口不能直接完成：

- eval；
- violation rate 统计；
- 多 seed 汇总；
- MuJoCo 或 didactic 训练；
- baseline 对比；
- 消融实验；
- 自动绘图与论文图表复现；
- 完整训练恢复。

---

## 15. 建议的后续工程化方向

若要把仓库整理为可复现实验框架，建议按以下顺序推进：

1. 建立统一环境协议，明确 `reset/step_con`、reward、constraint、state、observation 和 action shape。
2. 拆分 `train.py` 与 `eval.py`，实现 100-episode violation rate 和 task cost 统计。
3. 改为 state-dict checkpoint，保存所有网络、optimizer、参数和 RNG 状态。
4. 让 loss ablation 和 loss weight 成为真正可控参数。
5. 明确并统一 `z*` 算法，使代码、论文和实验设置一致。
6. 为 MuJoCo 和 didactic 编写 adapter，接入同一 agent API。
7. 为每个场景统一连续约束的符号和尺度。
8. 增加多 seed launcher、CSV 日志、聚合和绘图脚本。
9. 删除或归档重复和未使用的旧代码，保留清晰的实际执行路径。

---

## 16. 总结

本仓库的核心价值在于展示了一条较完整的 EPI 学习链：

```text
连续时间 rollout
  -> 学习动力学与任务/约束代价
  -> 学习 return/constraint critics
  -> 使用 epigraph HJB residual 与 VGI 约束 value 及其梯度
  -> 通过 learned Hamiltonian 更新 decentralized actors
```

MPE 的关键改造是引入外部 `dt`，用 `step_con()` 和 `integrate_state_continuous()` 替代固定时间步，并把任务 reward、离散碰撞惩罚和连续约束信号拆开。

与此同时，当前仓库保留了明显的研究迭代痕迹：部分模块未接入、若干参数仅为占位、不同场景约束语义不统一、eval 和完整 checkpoint 缺失。阅读和扩展代码时，应始终区分“论文描述”“仓库中存在的代码”和“当前主入口实际执行的代码”。
