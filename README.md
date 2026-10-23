# Lightweight DiT：CIFAR-10 50000 张从零训练

21.6M 参数的无条件 AdaLN-Zero DiT（patch=2, hidden=384, depth=8, heads=8, v-prediction + DDIM），
在完整 CIFAR-10 train split（50000 张 32×32）上双卡 DDP 从零训练 400k step。

## 结果

| 指标 | 数值 |
|---|---|
| 训练 | 400000 step 全程稳定，最终 loss 0.0939 |
| 相邻像素相关性 | 0.8806（真实数据 0.9344） |
| Laplacian RMS | 0.3285（真实数据 0.3877） |
| 样本多样性 | 16 图两两余弦相似度 mean 0.031，无模式重复 |

最终样本见 `dit_runs/lightweight_v4_50k_scratch_qknorm/samples/`。

## 核心要点：QK-norm 是必需的

无 QK-norm 版本两次在 ~315k/323k 步发散：q/k 范数缓增 → attention logit 爆炸 →
softmax 熵坍缩 → 输出塌缩为零预测，采样变纯噪声。加 RMSNorm(q)/RMSNorm(k) 后
400k 全程 logit 有界（10–46）、熵稳定，零发散。

完整问题定位与解决过程见 [TRAINING_ISSUES_AND_SOLUTIONS.md](TRAINING_ISSUES_AND_SOLUTIONS.md)。

## 文件

| 文件 | 说明 |
|---|---|
| `lightweight_dit.py` | 训练/采样脚本：DDP、QK-norm、稳定防护、EMA、DDIM |
| `analyze_samples.py` / `eval_grid.py` | 样本统计指标评估 |
| `attention_forensics*.py` | attention logit / 熵坍缩取证 |
| `forensic_checkpoints.py` / `module_diff.py` | 权重与模块漂移取证 |
| `inspect*_history.py` | loss 曲线与发散点分析 |
| `HANDOFF_50K_DIT_400K.md` | 实验交接文档 |

## 快速开始

```bash
# 双卡从零训练
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 lightweight_dit.py \
  --output-dir dit_runs/run1 --data-dir images_32x32_50000 \
  --train-steps 400000 --batch-size 128 --patch-size 2 \
  --hidden-size 384 --depth 8 --num-heads 8 --qk-norm \
  --prediction v --learning-rate 0.0003 --ema-decay 0.9999 --no-resume

# 采样（需指定 checkpoint）
python lightweight_dit.py --sample-only --qk-norm --prediction v \
  --checkpoint dit_runs/run1/checkpoint-400000.pt --train-steps 400000 \
  --patch-size 2 --hidden-size 384 --depth 8 --num-heads 8 \
  --sample-steps 100 --num-samples 64
```
