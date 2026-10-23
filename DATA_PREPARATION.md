# 数据获取与准备文档（CIFAR-10 50000 张 32×32 RGB）

## 1. 数据概况

| 项 | 值 |
|---|---|
| 数据集 | CIFAR-10 **train split**（不含测试集） |
| 数量 | 50000 张 |
| 格式 | 每张一张 `32×32 RGB PNG`，`00001.png ~ 50000.png` |
| 标签 | **不导出**（本实验是无条件生成，label 全部丢弃） |
| 磁盘占用 | 约 120MB |
| 输出目录 | `images_32x32_50000/` |

按 GitHub 仓库约定，**数据目录与 parquet 缓存均不上传**，用本文档 + `prepare_data.py` 现场重建。

## 2. 方式一（推荐，完全自包含）：torchvision 下载

```bash
# 容器内执行（宿主机无 PyTorch）
docker exec -it yhy_onnx bash -lc '
cd /nfs/AE/yhy/DiT_train && python prepare_data.py
'
```

行为：

1. torchvision 从官方源下载 CIFAR-10（约 170MB tar.gz，存到 `datasets_cache/`）；
2. 取 train split 50000 张，逐张转 PNG 字节；
3. 全量校验（数量 50000 / 尺寸 32×32 / 模式 RGB）后写出 `images_32x32_50000/`。

常用参数：

```bash
python prepare_data.py --out images_32x32_50000          # 自定义输出目录
python prepare_data.py --cache-dir datasets_cache        # 自定义下载缓存目录
```

## 3. 方式二：从已有 parquet 提取

如果已有 `datasets_cache/cifar10-train.parquet`（列 `img.bytes` = 原始 PNG 字节，
行数 50000）：

```bash
docker exec -it yhy_onnx bash -lc '
cd /nfs/AE/yhy/DiT_train && python prepare_data.py \
  --from-parquet datasets_cache/cifar10-train.parquet
'
```

> **已知坑**：早期交接文档示例代码写的是 `pf[:500].to_pandas(...)`，
> 这只会读前 500 行，随后 50000 的断言必然失败。正确写法是
> `pf.to_pandas(columns=["img.bytes"])` 读全表（`prepare_data.py` 已修正）。

## 4. 数据校验（准备完成后必做）

```bash
find images_32x32_50000 -maxdepth 1 -type f | wc -l   # 期望 50000
du -sh images_32x32_50000                             # 期望约 120M
```

`prepare_data.py` 内置校验（每张都验证 32×32 / RGB / PNG 头），脚本正常退出即数据合格。

## 5. 与训练的衔接

```bash
--data-dir images_32x32_50000   # 训练脚本只吃「PNG 目录」，不做任何数据集特判
```

- Dataset 实现：`ImageFolderTensor`（`lightweight_dit.py`），排序后逐张读取，
  Resize 到 32×32 + Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 到 [-1,1]；
- 双卡 DDP 时由 `DistributedSampler` 自动均分，全局 batch 256；
- 数据顺序每个 epoch 由 `sampler.set_epoch()` 重排。

## 6. 复现实验的完整顺序

```bash
python prepare_data.py                 # 1. 生成数据
torchrun --standalone --nproc_per_node=2 lightweight_dit.py \
  --output-dir dit_runs/run1 --data-dir images_32x32_50000 \
  --train-steps 400000 --batch-size 128 --patch-size 2 \
  --hidden-size 384 --depth 8 --num-heads 8 --qk-norm \
  --prediction v --learning-rate 0.0003 --ema-decay 0.9999 --no-resume   # 2. 训练
```

## 7. 上传约定（.gitignore）

```gitignore
images_32x32/
images_32x32_5000/
images_32x32_50000/
datasets_cache/
```

仓库只保留：数据准备脚本（`prepare_data.py`）、本文档、训练代码与最终
checkpoint（git-lfs）。任何环境 `git clone` 后跑一遍第 2 节命令即可完整复现。
