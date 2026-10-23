# DiT 从零训练交接文档：50000 张 CIFAR / 400000 step / GPU 6+GPU 7

更新时间：2026-10-21  
工作目录：`/nfs/AE/yhy/DiT_train`  
训练容器：`yhy_onnx`

> 当前没有正在运行的 DiT 训练进程。新实验要求从零开始，不要 resume 旧 checkpoint。

## 1. 新实验目标

- 数据：完整 CIFAR-10 train split，共 50000 张
- 数据格式：每张 `32x32 RGB PNG`，不导出标签
- 训练方式：从零开始，`--no-resume`
- 训练步数：`400000` optimizer steps
- GPU：宿主机 GPU 6 + GPU 7
- 分布式：PyTorch DDP / NCCL，每卡一个进程
- Batch：建议 `128 / GPU`，全局 batch `256`
- 输出目录：`dit_runs/lightweight_v4_50k_scratch_ddp`

模型结构：

```text
image_size = 32
patch_size = 2
in_channels = 3
hidden_size = 384
depth = 8
num_heads = 8
mlp_ratio = 4
time_dim = 256
tokens = 256
parameters ≈ 21.62M
```

## 2. 当前状态

必须使用 `yhy_onnx` 容器；宿主机默认 Python 没有 PyTorch。

容器环境：

```text
Python 3.11.10
torch 2.5.1+cu121
torchvision 0.20.1+cu121
fastparquet 已安装
```

当前关键文件：

```text
images_32x32/                         # 500 张，早期实验
images_32x32_5000/                    # 5000 张，上一个实验
datasets_cache/cifar10-train.parquet  # 115MB，完整 50000 张 CIFAR train split
lightweight_dit.py                    # 当前为单 GPU 脚本，尚未支持 DDP
dit_runs/lightweight_v4/              # 旧实验，约 5.9GB
```

旧模型：

```text
dit_runs/lightweight_v4/latest.pt
dit_runs/lightweight_v4/checkpoint-200000.pt
```

两者都是 step `200000`。该模型不是从零在 50000 张上训练的：

- step `0-40000`：主要在 500 张数据上训练；
- step `40000-47000`：切换到 5000 张后中断；
- step `47000-200000`：在 5000 张上继续训练。

它只作为参考，不要作为新实验 resume 起点。

最后检查时：

```text
无 lightweight_dit.py / torchrun 进程
dit_runs/lightweight_v4/training.lock 不存在
GPU 6 / GPU 7 均空闲
```

## 3. 提取完整 50000 张图像

目标目录：

```text
images_32x32_50000/
```

Parquet 中 `img.bytes` 是原始 PNG 字节，`label` 字段不要导出。

执行：

```bash
docker exec yhy_onnx bash -lc '
cd /nfs/AE/yhy/DiT_train && python - <<"PY"
from pathlib import Path
from io import BytesIO
import fastparquet
from PIL import Image

OUT = Path("images_32x32_50000")
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f"Refusing to modify non-empty directory: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)

pf = fastparquet.ParquetFile("datasets_cache/cifar10-train.parquet")
df = pf[:500].to_pandas(columns=["img.bytes"])
raws = df["img.bytes"].tolist()
if len(raws) != 50000:
    raise RuntimeError(f"Expected 50000 images, got {len(raws)}")

for count, raw in enumerate(raws, 1):
    with Image.open(BytesIO(raw)) as image:
        image.load()
        if image.size != (32, 32):
            raise RuntimeError((count, image.size))
        if image.mode != "RGB":
            raise RuntimeError((count, image.mode))
    (OUT / f"{count:05d}.png").write_bytes(raw)

files = sorted(OUT.glob("*.png"))
if len(files) != 50000:
    raise RuntimeError((len(files), OUT))
print(f"Saved and validated {len(files)} images to {OUT.resolve()}")
PY
'
```

检查：

```bash
find images_32x32_50000 -maxdepth 1 -type f | wc -l
du -sh images_32x32_50000
```

预期数量 `50000`，大小约 `120MB`。

## 4. 修改 `lightweight_dit.py` 支持 DDP

当前脚本仍是单 GPU 版本。直接用 torchrun 启动不可靠，必须先修改。

### 4.1 新增 import

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

### 4.2 初始化进程组和 device

在 `main()` 开头加入：

```python
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
distributed = world_size > 1

if distributed:
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device(args.device)
```

启动时设置 `CUDA_VISIBLE_DEVICES=6,7` 后：

```text
cuda:0 -> 宿主机 GPU 6
cuda:1 -> 宿主机 GPU 7
```

不要让两个 rank 都写死使用 `cuda:0`。

### 4.3 包装 DDP

建议保留底层 `model` 引用：

```python
model = LightweightDiT(config).to(device)
ema_model = LightweightDiT(config).to(device)

training_model = model
if distributed:
    training_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
```

训练 forward 使用：

```python
prediction = training_model(x_t, t)
```

EMA、checkpoint、采样继续使用底层 `model` / `ema_model`，避免 state dict 出现 `module.` 前缀。

### 4.4 DistributedSampler

将当前 DataLoader 的 `shuffle=True` 改为 sampler 模式：

```python
sampler = None
if distributed:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

每个 epoch 调用：

```python
if sampler is not None:
    epoch += 1
    sampler.set_epoch(epoch)
```

`--batch-size 128` 表示每卡 128，全局 batch 256。

### 4.5 rank 0 独占 IO

只有 rank 0 允许：

- 创建/删除 `training.lock`
- 写 `config.json`
- 写 `history.json`
- 保存 `latest.pt`
- 保存 milestone checkpoint
- 生成 preview / final samples
- 打印训练日志

checkpoint 保存后可用 `dist.barrier()` 同步。其他 rank 不要写同一 NFS 文件。

### 4.6 训练细节

梯度裁剪继续针对底层模型：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

只有 rank 0 需要 `loss.item()`、打印日志和写 history。结束时：

```python
if distributed:
    dist.destroy_process_group()
```

## 5. DDP 冒烟测试

先跑 20 step，不要直接启动 400000 step：

```bash
docker exec yhy_onnx bash -lc '
cd /nfs/AE/yhy/DiT_train && \
CUDA_VISIBLE_DEVICES=6,7 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
torchrun --standalone --nproc_per_node=2 lightweight_dit.py \
  --output-dir /tmp/dit-ddp-smoke \
  --data-dir images_32x32_5000 \
  --train-steps 20 \
  --batch-size 16 \
  --patch-size 2 \
  --hidden-size 384 \
  --depth 8 \
  --num-heads 8 \
  --prediction v \
  --num-timesteps 400 \
  --sample-steps 4 \
  --num-samples 4 \
  --no-resume \
  --seed 2027
'
```

必须确认：

1. world size 是 2；
2. 宿主机 GPU 6 和 GPU 7 都有显存/进程占用；
3. rank 0 能保存输出；
4. 进程正常退出；
5. `/tmp/dit-ddp-smoke/training.lock` 被删除。

## 6. 正式从零训练命令

冒烟测试通过后执行：

```bash
docker exec yhy_onnx bash -lc '
cd /nfs/AE/yhy/DiT_train && \
CUDA_VISIBLE_DEVICES=6,7 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=2 lightweight_dit.py \
  --output-dir dit_runs/lightweight_v4_50k_scratch_ddp \
  --data-dir images_32x32_50000 \
  --train-steps 400000 \
  --batch-size 128 \
  --patch-size 2 \
  --hidden-size 384 \
  --depth 8 \
  --num-heads 8 \
  --prediction v \
  --num-timesteps 400 \
  --sample-steps 100 \
  --num-samples 64 \
  --learning-rate 0.0003 \
  --warmup-steps 1000 \
  --ema-decay 0.9999 \
  --checkpoint-interval 20000 \
  --latest-checkpoint-interval 5000 \
  --preview-interval 20000 \
  --no-resume \
  --seed 2027
'
```

关键点：

- 必须使用 `--no-resume`
- 必须使用新输出目录
- 必须通过 `CUDA_VISIBLE_DEVICES=6,7` 绑定两张卡
- `torchrun --nproc_per_node=2`

建议在 Codex PTY 中前台运行，便于观察日志。如果后台运行，必须记录 PID 和日志文件。

## 7. 预计耗时与空间

旧单卡速度约 `10-13 step/s`。双卡 DDP 不一定线性加速，估计 `15-22 step/s`。  
400000 step 预计 `5-8 小时`。

当前每个 v4 checkpoint 约 `346MB`。每 20000 step 保存一个 milestone：

```text
20 × 346MB ≈ 6.9GB
```

NFS 剩余空间约 8.7TB，空间充足。

## 8. 监控命令

查看 GPU 6/7：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits | awk -F, '$1 ~ /^ *[67]$/ {print}'
```

查看训练进程：

```bash
ps -ef | grep -E '[l]ightweight_dit|[t]orchrun'
```

读取最新 step：

```bash
docker exec yhy_onnx bash -lc '
cd /nfs/AE/yhy/DiT_train && python - <<"PY"
import torch
from pathlib import Path
p = Path("dit_runs/lightweight_v4_50k_scratch_ddp/latest.pt")
if not p.exists():
    print("latest.pt missing")
else:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("step:", ck["step"])
    print("config:", ck["config"])
PY
'
```

## 9. 完成后采样

训练完成后用单卡采样，建议使用 GPU 7：

```bash
docker exec yhy_onnx bash -lc '
cd /nfs/AE/yhy/DiT_train && \
CUDA_VISIBLE_DEVICES=7 \
PYTHONUNBUFFERED=1 \
python lightweight_dit.py \
  --output-dir dit_runs/lightweight_v4_50k_scratch_ddp \
  --device cuda:0 \
  --prediction v \
  --sample-only \
  --resume \
  --train-steps 400000 \
  --patch-size 2 \
  --hidden-size 384 \
  --depth 8 \
  --num-heads 8 \
  --sample-steps 100 \
  --num-samples 64 \
  --seed 2027
'
```

必须显式传 `--train-steps 400000`，否则 sample-only 的输出文件名可能使用默认 `6000`，虽然权重是 latest，但文件名会误导。

预期输出：

```text
dit_runs/lightweight_v4_50k_scratch_ddp/samples/final_step400000_grid.png
dit_runs/lightweight_v4_50k_scratch_ddp/samples/final_step400000_01.png
...
dit_runs/lightweight_v4_50k_scratch_ddp/samples/final_step400000_16.png
```

## 10. 质量评估

比较真实 50000 张、旧 v4 step 200000、新模型 step 200000/400000：

- 平均亮度
- 标准差
- 相邻像素相关性
- Laplacian 边缘能量
- 样本多样性
- 是否模式重复

参考指标函数：

```python
def metrics(x):
    x = x.float().cuda()
    lap = (
        -4 * x[:, :, 1:-1, 1:-1]
        + x[:, :, :-2, 1:-1]
        + x[:, :, 2:, 1:-1]
        + x[:, :, 1:-1, :-2]
        + x[:, :, 1:-1, 2:]
    )
    y = x.roll(-1, dims=3)
    a = x - x.mean()
    b = y - y.mean()
    corr = torch.dot(a.flatten(), b.flatten()) / (
        a.flatten().norm() * b.flatten().norm() + 1e-12
    )
    return x.mean().item(), x.std().item(), corr.item(), lap.pow(2).mean().sqrt().item()
```

旧 5000 张模型 step 200000 的 16 张单独生成图参考：

```text
mean ≈ 0.4729
std ≈ 0.1313
adjacent corr ≈ 0.9140
Laplacian RMS ≈ 0.1234
```

## 11. 已知坑

1. 不要在宿主机直接训练，必须进入 `yhy_onnx`。
2. 新实验必须 `--no-resume`。
3. 当前 `lightweight_dit.py` 尚未支持 DDP，必须先修改。
4. 双卡时每个 rank 使用自己的 `LOCAL_RANK`。
5. 只有 rank 0 写文件。
6. 保存图像时已手动 `(x + 1) * 0.5`，不要改回只依赖 torchvision `value_range`。
7. sample-only 必须显式传 `--train-steps 400000`。
8. 如果中断，先清理残留 torchrun/worker 进程，再删除 stale `training.lock`。

## 12. 下一个会话最短执行清单

1. 提取并校验 `images_32x32_50000/`，数量必须为 50000。
2. 给 `lightweight_dit.py` 增加 NCCL DDP、DistributedSampler、rank0-only IO、LOCAL_RANK device。
3. 用 GPU 6+7 跑 20 step 冒烟测试。
4. 启动 `400000 step` 从零训练。
5. 完成后单卡采样并评估。
