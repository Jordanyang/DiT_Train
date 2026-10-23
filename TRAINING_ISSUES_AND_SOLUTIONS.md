# DiT 50000 张 CIFAR 从零训练：问题记录与解决方案

时间：2026-10-21 ~ 2026-10-23
工作目录：`/nfs/AE/yhy/DiT_train`
容器：`yhy_onnx`（Python 3.11 / torch 2.5.1+cu121）
硬件：宿主机 GPU 6 + GPU 7（RTX 4090 ×2）

---

## 0. 实验最终结果（先说结论）

| 版本 | 结局 | 最终 loss | 样本质量 |
|---|---|---|---|
| v1 无 QK-norm（seed 2027） | 300k 后发散，400k 权重作废 | 0.56 | 320k 起纯噪声 |
| v1 续跑（降 LR + 尖峰防护） | 323k 再次发散 | 0.96 | 330k 起纯噪声 |
| **v2 加 QK-norm（seed 2028）** | **400k 全程稳定** | **0.0939** | **结构完整、多样** |

最终保留产物（其余中间权重已清理）：

```text
dit_runs/lightweight_v4_50k_scratch_qknorm/checkpoint-400000.pt  # 331M
dit_runs/lightweight_v4_50k_scratch_qknorm/config.json
dit_runs/lightweight_v4_50k_scratch_qknorm/history.json
dit_runs/lightweight_v4_50k_scratch_qknorm/samples/               # final grid + 16 单图
```

最终指标（64 样本，100 步 DDIM，EMA）：

| 指标 | v2 400k | 真实数据 |
|---|---|---|
| mean | -0.0099 | -0.0527 |
| std | 0.3227 | 0.5347 |
| 相邻像素相关性 | 0.8806 | 0.9344 |
| Laplacian RMS | 0.3285 | 0.3877 |
| 16 张单图两两余弦相似度 | mean 0.031 / max 0.59（无模式重复） | — |

---

## 1. 问题一：交接文档的 parquet 提取代码只取了 500 行

**现象**：交接文档示例代码写的是 `pf[:500].to_pandas(...)`，但随后断言 `len(raws) == 50000`，直接照抄必然报错。

**解决**：改为 `pf.to_pandas(columns=["img.bytes"])` 读全表，逐张校验 32×32 RGB PNG 后写出。最终 50000 张 / 120MB，与 parquet 行数一致。

**教训**：交接文档里的代码示例也可能有笔误，执行前先小步验证（先查 `pf.count()`、列名、首行字节头）。

---

## 2. 问题二：训练脚本不支持 DDP

**现象**：原始 `lightweight_dit.py` 是单 GPU 脚本，直接 torchrun 启动不可靠（device 写死 `cuda:0`、无进程组、无 rank 区分 IO）。

**解决**：改造为标准 DDP 脚本，要点：

1. `dist.init_process_group("nccl")`，device 用 `cuda:{LOCAL_RANK}`；`CUDA_VISIBLE_DEVICES=6,7` 下 cuda:0/cuda:1 分别映射宿主机 GPU6/7。
2. 训练 forward 用 `DDP(model)` 包装的 `training_model`；EMA、checkpoint、采样用底层 `model`，保证 state dict 无 `module.` 前缀。
3. `DistributedSampler(shuffle=True, drop_last=True)`，每个 epoch `sampler.set_epoch(epoch)`。
4. 只有 rank 0 做所有 IO（lock/config/history/latest/milestone/preview/日志），写完 `dist.barrier()` 同步。
5. **preview 和最终采样前后都要 barrier**：rank0 采样期间其他 rank 若继续跑，会在 DDP allreduce 上等待，NCCL 超时可能报错；显式 barrier 让所有 rank 一起等。
6. 每个 rank 用 `seed + rank` 的独立 RNG 流（模型初始化仍一致，因为 DDP 会广播 rank0 权重）。

**验证**：20 step 双卡冒烟：world_size=2、exit 0、rank0 产物齐全、`training.lock` 删除、checkpoint 无 `module.` 前缀。

---

## 3. 问题三：400k 训练在 ~315k 步后发散（第一次）

**现象**：

- loss 从稳定 0.094 在 step ≈ 315000–317400 开始爬升，一路涨到 0.56 后不再恢复；
- 预览样本从 320000 起变成纯噪声（相邻像素相关性 0.88 → 0，Laplacian RMS 0.34 → 3.2+）；
- 400000 的最终权重不可用。

**定位过程**：

1. loss 曲线二分定位：`history.json` 找到第一个 >0.2 的点在 317400，发散起点 ~315k；
2. 重新采样 260k/280k/300k 三个 milestone：全部正常（corr ≈ 0.883），确认 **checkpoint-300000 是最后一个干净存档**；
3. 判断为长跑训练不稳定（bf16 + AdamW 损失尖峰），当时先采取了「降 LR + 稳定性防护 + 从 300k 续跑」的方案。

---

## 4. 问题四：续跑加了防护后吞吐暴跌（临时误判）

**现象**：第一版防护是每步做一次 guard loss 的 `all_reduce` + 双 rank `.item()` 同步，启动后速度从 12.8 掉到 ~1.3 steps/s，GPU7 利用率 0%。

**排查**：多采样几次 GPU 利用率 + 观察后续日志，发现 1 分钟后恢复到 ~11.5 steps/s——真实原因是 **50000 个文件的 dataloader 首轮预热**，防护本身的稳定代价约为 10%。

**仍做的优化**（后来证明有价值）：

1. loss 尖峰检查从每步一次改为每 10 步一次（`--guard-interval`），跨 rank 平均后决策保证一致；
2. NaN/Inf 防护改为**免同步**方案：每步 `grad.nan_to_num_()`（GPU 端异步）+ `clip_grad_norm_`，inf 情况由 clip 自然压成零更新。

**关键教训（DDP）**：任何「跳过 optimizer step」的分支必须保证**所有 rank 做出相同决策**（用 all_reduce 后的全局 loss 判断），否则各 rank 前向次数错位，NCCL 死锁。

---

## 5. 问题五：进程清理三次才清干净

**现象**：想停掉续跑重启时：

1. PTY 里 Ctrl-C 只杀掉了外层 `docker exec`，容器内 torchrun + worker 继续跑；
2. 宿主机 `kill -9` 无效（进程属 root，yhy 用户无权限，报错被 `2>/dev/null` 吞掉）；
3. 容器内 `pkill -f "python3.11 -u lightweight_dit.py"` 把**自己的 bash 也杀了**（`-f` 匹配到了自身命令行），但目标进程还活着。

**最终解决**：容器内先 `ps` 确认 PID，再 `kill -9 <具体PID>`（父 torchrun 死后 worker 成为孤儿进程，需要单独杀），最后确认 GPU 显存归零 + 删除 stale `training.lock`。

**正确套路**：

```bash
docker exec yhy_onnx bash -lc 'ps -ef | grep lightweight_dit | grep -v grep'  # 找容器内 PID
docker exec yhy_onnx bash -lc 'kill -9 <PID...>'
docker exec yhy_onnx bash -lc 'rm -f <output_dir>/training.lock'
nvidia-smi ...  # 确认显存释放
```

---

## 6. 问题六：降 LR 续跑仍在 ~323k 再次发散（根因不是 LR）

**现象**：

- 从 300k 续跑，LR 减半（实际区间 ~3.1e-5 → 1e-5，原发散区间 5.2e-5 → 3.9e-5）；
- 健康（loss 0.0943）运行到 **step 322000**，323000 起缓升，327800 破 0.15，最终 ~0.96（≈ 零预测平凡解）；
- 尖峰防护全程 skipped=0——**这次是缓变塌缩，不是尖峰**，6×中位数阈值天然拦不住缓慢爬坡。

**关键推理**：两次发散点高度重合（~315k / ~323k），且第二次 LR 已低至 2.2e-5 仍塌缩 → **不是学习率问题，而是与步数相关的结构性漂移**。

---

## 7. 问题七：系统性取证定位真正的根因

为避免盲目重试，写了三组取证脚本（保留在仓库中）：

| 脚本 | 作用 |
|---|---|
| `forensic_checkpoints.py` | 跨 checkpoint 对比参数范数、bf16/fp32 前向输出幅度与 MSE |
| `module_diff.py` | 按模块计算 checkpoint 间的相对权重漂移 |
| `attention_forensics.py` / `attention_forensics_qk.py` | 各 block 的 attention logit 上限、q/k 范数、softmax 熵 |

**逐一排除的假设**：

| 嫌疑 | 证据 | 结论 |
|---|---|---|
| bf16 精度问题 | fp32 与 bf16 前向输出几乎逐位一致 | 排除 |
| 权重爆炸 | paramL2 420→423 平稳，maxW 稳定 1.57 | 排除 |
| 学习率过高 | 2.2e-5 照样塌缩，且两次塌缩点重合 | 排除 |

**真正发现的现象（attention 取证，决定性证据）**：

| step | block1 q_norm | block1 logit_max | attention 熵 |
|---|---|---|---|
| 280k（健康） | 4.05 | 182 | 4.92 |
| 320k（loss 尚正常） | 4.90 | **1329** | 4.93 |
| 330k（开始塌） | **3572** | **2.2e7** | **0.003** |
| 400k（全坏） | 8387 | 9.2e7 | ~0 |

同时 `module_diff.py` 显示漂移最大的是 `time_mlp`（20–37%）和 `blocks.1/6/7.ada_ln_mlp`（27–32%）。

**根因链条**：

```text
q/k 范数随训练缓慢单调增长（与 LR 无关）
  → attention logit 爆炸（320k 时 block1 已 1329）
  → softmax 熵坍缩成 one-hot（330k 时 0.003）
  → 梯度饥饿，time_mlp / ada_ln 被迫大幅补偿漂移
  → 模型输出幅度塌缩向零预测（MSE→1.0，采样=纯噪声）
```

即：**AdaLN-Zero DiT 无 QK-norm 时，长训练下 attention 熵坍缩是必然的时间炸弹**；loss 在 300k 前看似平稳只是因为病根（logit 增长）尚未越过临界点。

---

## 8. 问题八（最终修复）：加 QK-norm 后 400k 全程稳定

**修复**：给每个 block 的 q、k 各接一个 RMSNorm（ViT-22B / SD3 / Flux 同款做法），把 logit 幅度钉死在源头：

```python
# DiTConfig 增加 qk_norm 字段 + CLI --qk-norm（默认 False，兼容旧 checkpoint）
self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
# forward 中 reshape 到 (b, heads, tokens, head_dim) 后：
q = self.q_norm(q)
k = self.k_norm(k)
attention = F.scaled_dot_product_attention(q, k, v)
```

**为什么必须从零重训而不能续跑改造**：QK-norm 引入新参数；且旧权重（如 320k）的 attention 已在病态轨道上（block1 logit 1329、block4 熵 0.32），强行归一化会彻底改变其注意模式。

**v2 运行结果（2026-10-22 16:41 → 10-23 04:59，约 12.3h）**：

- loss 全程 0.092–0.095，最终 0.0939，min 0.0908 @ 326.8k，**零发散、零跳步**；
- attention 取证（100k/300k/400k 三个节点）：post-norm logit_max 稳定在 10–46，原始 q/k 范数 7–29 几乎不随训练变化，熵 100k ≈ 400k（block4: 1.76→1.87，是稳定特性而非坍缩）；对比 v1 塌缩时 logit 2×10⁷ / q_norm 8000+，证明 QK-norm 从机制上消除了熵坍缩；
- 样本指标各 milestone 全程平稳（corr 0.88–0.89），无中途劣化；
- 代价：吞吐 12.8 → ~8.5-9.4 steps/s（约 -30%，归一化打破 SDPA 融合路径），400k 多花约 2h，完全可接受。

---

## 9. 沉淀的工程组件与用法

**训练脚本新增能力**：

```bash
# 从零（v2 配置）
torchrun --standalone --nproc_per_node=2 lightweight_dit.py ... --qk-norm --no-resume
# 从指定 milestone 采样（latest.pt 已删，必须显式指定）
python lightweight_dit.py --sample-only --qk-norm \
  --checkpoint dit_runs/lightweight_v4_50k_scratch_qknorm/checkpoint-400000.pt \
  --train-steps 400000 --sample-steps 100 --num-samples 64 --seed 2028
```

新增 CLI：`--qk-norm`、`--checkpoint <path>`、`--loss-spike-factor`、`--guard-interval`。

**稳定性防护（保留在脚本中）**：

1. 每 `guard-interval` 步一次的全局 loss 检查（非有限或 > 中位数 × 阈值 → 全 rank 一致跳过该步）；
2. 每步异步 `grad.nan_to_num_()` + clip（防 NaN/Inf 污染权重与 EMA，零同步开销）。

**评估工具链**：

| 脚本 | 用途 |
|---|---|
| `analyze_samples.py` / `eval_grid.py` | 把 grid 还原成单图，计算 mean/std/相邻相关性/Laplacian RMS，对比真实数据 |
| `inspect_history.py` 系列 | loss 曲线扫描、发散点定位、跳步统计 |
| `forensic_checkpoints.py` | 参数范数 + bf16/fp32 前向对比 |
| `module_diff.py` | 按模块的跨 checkpoint 权重漂移 |
| `attention_forensics*.py` | attention logit / q,k 范数 / 熵坍缩检测 |

---

## 10. 经验教训清单

1. **长跑扩散模型训练必须防 attention 熵坍缩**：无 QK-norm 的 DiT 在几十万步尺度上等于定时炸弹，且 loss 平稳不等于健康（320k 时 logit 已 1329 但 loss 仍 0.094）。
2. **loss 曲线正常 ≠ 训练健康**：attention logit / 熵这类内部统计量才是早期预警指标，值得在 preview 间隔里顺手记录。
3. **发散分析要先取证再行动**：第一次失败后如果直接取证，可以省掉一次 2.5h 的无效续跑；「降 LR + 尖峰防护」只对尖峰型发散有效，对缓变漂移无效。
4. **DDP 中任何条件跳步必须全 rank 决策一致**，否则 NCCL 死锁；同步类防护要控制频率，避免与梯度 allreduce 串流影响吞吐。
5. **容器内进程清理要按容器内 PID 杀**；`pkill -f` 小心匹配到自身命令行。
6. **milestone 密度要足够**：第一次发散区间 300k–320k 只有 20k 粒度，最后干净点恰好是 300k；续跑时改成 10k 粒度后定位成本低很多。
7. **EMA 会被坏权重污染**：权重塌缩后 EMA 一同失效，无法靠 EMA 抢救，必须在权重层面解决。
8. **改动前留冒烟测试**：每次改脚本（DDP、防护、QK-norm）都先跑 20–30 step 小模型冒烟，全程零踩坑启动失败。

---

## 11. 当前磁盘状态

```text
dit_runs/lightweight_v4_50k_scratch_qknorm/   335M   # 只保留 checkpoint-400000.pt + 配置/记录/样本
dit_runs/lightweight_v4_50k_scratch_ddp/      ~7GB   # v1 病态 run（300k 前 milestone 可用但已无保留价值）
dit_runs/lightweight_v4_50k_scratch_ddp_continue/ ~3.6GB  # v1 续跑（全部病态）
dit_runs/lightweight_v4/                     ~5.9GB # 更早的 5000 张实验
images_32x32_50000/                          120M   # 数据集
```
