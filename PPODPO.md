## PPO（Proximal Policy Optimization）介绍

### 1. 核心思想：

PPO（近端策略优化）是一种经典的强化学习（RL）算法，其目标是通过与环境交互，逐步优化策略（policy），使其能获得最大累计回报（reward）。

PPO的特点是通过 **裁剪（clipping）策略更新** 的方式限制每次策略更新的幅度，确保训练过程稳定。

---

### 2. PPO的优化目标：

PPO最大化以下目标函数：

$$
J_{\text{PPO}}(\theta) = \mathbb{E}_{s,a \sim \pi_{\theta_{\text{old}}}}\left[
\min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \hat{A}(s,a),\;
\text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\epsilon, 1+\epsilon\right)\hat{A}(s,a)
\right)
\right]
$$

其中：

- \\(\pi_{\theta}(a|s)\\)：当前策略概率
- \\(\pi_{\theta_{\text{old}}}(a|s)\\)：旧策略概率
- \\(\hat{A}(s,a)\\)：优势函数（advantage）
- \\(\epsilon\\)：策略裁剪范围（如 0.2）

---

### 3. RLHF中的PPO训练流程：

1. **采样阶段**：
   - 使用 actor 策略生成对话（prompt + response）
   - 使用 reward 模型对生成结果评分
   - 使用 reference model 计算 KL 约束

2. **计算奖励和优势**：
   - 得到即时奖励（reward - KL）
   - 使用 critic 网络估计状态价值，计算优势函数（GAE）

3. **更新阶段**：
   - 使用新旧策略对比与优势，计算 PPO 损失
   - 更新 actor 和 critic 网络

---

### 4. PPO用的数据类型：

- 基于交互式数据（rollout）
- 需要外部 reward signal
- 需要 critic 辅助价值估计

---

## DPO（Direct Preference Optimization）介绍

### 1. 核心思想：

DPO 是一种监督学习方法，它利用人类偏好对（A优于B），直接优化策略网络，使其输出更符合人类的判断。

DPO不需要显式训练 reward 模型，而是将偏好信息直接内嵌进策略优化过程中。

---

### 2. DPO的优化目标：

DPO优化的目标是最大化：

$$
P_{\theta}(A \succ B) = \frac{\exp(\beta r_{\theta}(A))}{\exp(\beta r_{\theta}(A)) + \exp(\beta r_{\theta}(B))}
$$

其中：

- \\(r_{\theta}(x)\\)：策略网络对样本 \\(x\\) 的“打分”
- \\(\beta\\)：温度参数，控制对偏好敏感程度

---

### 3. DPO训练流程：

1. **偏好数据**：
   - 提供已标注的（A 优于 B）数据对

2. **模型推理**：
   - 对 A/B 同时输入模型，获得 reward score（打分）

3. **优化目标**：
   - 直接优化 A 胜于 B 的概率

---

### 4. DPO用的数据类型：

- 不需要环境交互
- 只需要人类偏好数据对（A优于B）
- 不需要 critic 网络

---

## PPO vs DPO：关键区别对比

| 比较维度        | PPO（Proximal Policy Optimization） | DPO（Direct Preference Optimization） |
|----------------|--------------------------------------|----------------------------------------|
| 本质           | 强化学习（RL）                       | 监督学习（SL + 偏好学习）              |
| 是否需要 reward 模型 | ✅ 是                              | ❌ 否                                   |
| 是否需要 Critic 网络 | ✅ 是                              | ❌ 否                                   |
| 使用数据类型    | 交互式生成数据（prompt + response） | 标注偏好数据对（A 优于 B）             |
| 是否计算 Advantage | ✅ 是                              | ❌ 否                                   |
| 更新方式        | 策略梯度 + PPO裁剪                   | 偏好概率最大化                          |
| 收敛速度        | 中等                                 | 快                                     |
| 稳定性          | 可能不稳定（超参数敏感）             | 通常较稳定                             |
