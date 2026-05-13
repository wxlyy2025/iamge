# MineralGANAugmentInversion

这是一个独立的新工程，用于验证“矿预测样本太少时，先用 GAN 生成训练样本 patch，再用增强样本训练成矿远景反演模型”的流程。

它不会修改原项目已有的 `data/demo`、`runs/demo_*` 等内容。默认输出都写入：

```text
gan_augmented_inversion/workspace/
```

## 工程逻辑

1. 重新生成一份少矿点合成数据：
   - `evidences.npy`：多源找矿证据图层，形状为 `[C, H, W]`
   - `deposits.csv`：少量矿点坐标
   - `true_prospectivity.npy/png`：仅用于 demo 验证的真实远景场

2. 训练样本扩充 GAN：
   - 从少量矿点附近裁剪真实正样本 patch
   - GAN 学习生成“证据图层 patch + 标签 patch”
   - 输出 `generated_patches.npz`

3. 训练反演模型：
   - 混合真实 patch 和 GAN 生成 patch
   - 使用 `resunet` 等模型做成矿远景概率反演
   - 输出 `checkpoint_final.pt`

4. 全图预测：
   - 输出 `prospectivity.npy/png`
   - 输出 `uncertainty.npy/png`

## 一键运行

使用自动生成的少样本 demo 数据：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py --device cpu
```

如果有可用 CUDA：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py --device cuda
```

## 换成自己的数据

你不需要改代码，直接传入自己的证据图层和矿点 CSV：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py ^
  --device cpu ^
  --evidence D:\your_data\evidences.npy ^
  --deposits D:\your_data\deposits.csv ^
  --work-dir gan_augmented_inversion\workspace_your_data
```

如果只是跑 demo，但想改变自动生成的矿点数量，可以用：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py ^
  --device cpu ^
  --demo-deposits 12
```

证据图层要求：

```text
evidences.npy shape = [C, H, W]
```

其中：

- `C` 是证据图层数量，例如地球化学、断裂、岩性、物探等。
- `H, W` 是所有图层统一后的高度和宽度。
- 所有图层必须已经配准到同一网格。

矿点 CSV 要求：

```csv
row,col,deposit
42,38,1
88,90,1
```

也支持列名写成：

```csv
y,x,deposit
42,38,1
88,90,1
```

如果你的证据图层是多个 `.tif`，可以把它们放到同一个文件夹里，然后把文件夹路径传给 `--evidence`：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py ^
  --device cpu ^
  --evidence D:\your_data\evidence_tifs ^
  --deposits D:\your_data\deposits.csv ^
  --work-dir gan_augmented_inversion\workspace_your_tif_data
```

## 关键输出

```text
gan_augmented_inversion/workspace/
  data_few_shot/
    evidences.npy
    deposits.csv
    few_shot_truth.png
  sample_gan/
    sample_gan.pt
    generated_patches.npz
    generated_patches_preview.png
  augmented_inversion/
    checkpoint_final.pt
    history.json
  prediction/
    prospectivity.png
    uncertainty.png
  manifest.json
```

## 重要说明

这个工程和原项目里的 GAN 用法不同：

- 原项目 GAN：生成器就是反演网络，判别器用于约束预测图的空间真实性。
- 本工程 GAN：先训练一个样本生成 GAN，生成更多训练 patch；再把这些 patch 加入反演模型训练。

因此本工程明确实现了“GAN 数据扩充 + 模型反演”。
