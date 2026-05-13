# 使用自己的矿预测数据

这份说明只针对新工程 `gan_augmented_inversion`。你不需要修改原项目数据，也不需要手动改 Python 代码。

## 1. 你需要准备什么

准备两个输入：

```text
1. 证据图层
2. 矿点 CSV
```

证据图层推荐保存为：

```text
evidences.npy
```

形状必须是：

```text
[C, H, W]
```

含义：

- `C`：证据图层数量，比如 Cu、Mo、断裂距离、岩性、磁异常等。
- `H`：图像高度。
- `W`：图像宽度。

矿点文件格式：

```csv
row,col,deposit
42,38,1
88,90,1
```

其中：

- `row` 是矿点所在像素行号。
- `col` 是矿点所在像素列号。
- `deposit` 可以都写 1，当前代码主要使用 row/col。

如果你的列名是 `y,x` 也可以：

```csv
y,x,deposit
42,38,1
88,90,1
```

## 2. 直接运行自己的数据

示例：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py ^
  --device cpu ^
  --evidence D:\your_data\evidences.npy ^
  --deposits D:\your_data\deposits.csv ^
  --work-dir gan_augmented_inversion\workspace_your_data
```

如果你的电脑能用 CUDA：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py ^
  --device cuda ^
  --evidence D:\your_data\evidences.npy ^
  --deposits D:\your_data\deposits.csv ^
  --work-dir gan_augmented_inversion\workspace_your_data
```

## 3. 如果你的证据图层是 tif

把所有已经配准、同尺寸的 `.tif` 放到一个文件夹，例如：

```text
D:\your_data\evidence_tifs\
  01_cu.tif
  02_mo.tif
  03_fault.tif
  04_lithology.tif
```

然后运行：

```powershell
D:\anaconda\envs\deep\python.exe gan_augmented_inversion\run_pipeline.py ^
  --device cpu ^
  --evidence D:\your_data\evidence_tifs ^
  --deposits D:\your_data\deposits.csv ^
  --work-dir gan_augmented_inversion\workspace_your_tif_data
```

程序会按文件名排序读取 tif，所以建议文件名前面加 `01_`、`02_` 这种顺序编号。

## 4. 常用参数怎么改

矿点很少时，建议先用这些参数：

```powershell
--patch-size 32
--generated-samples 256
--generated-fraction 0.5
--positive-weight 50
```

含义：

- `--patch-size`：训练 patch 大小。矿点少、图幅小可以用 32；图幅大可以试 64。
- `--generated-samples`：GAN 生成多少个增强 patch。
- `--generated-fraction`：反演训练时生成样本占比，0.5 表示约一半来自 GAN。
- `--positive-weight`：正样本权重，矿点越少可以越大，一般 30 到 80。

如果只是运行 demo 数据，想调整自动生成的矿点数量，用：

```powershell
--demo-deposits 12
```

更正式训练可以增加轮数：

```powershell
--gan-epochs 50
--inversion-epochs 80
--samples-per-epoch 2048
```

先小规模跑通，再加大参数。

## 5. 输出在哪里

如果你设置：

```powershell
--work-dir gan_augmented_inversion\workspace_your_data
```

输出会在：

```text
gan_augmented_inversion/workspace_your_data/
  sample_gan/
    generated_patches.npz
    generated_patches_preview.png
    sample_gan.pt
  augmented_inversion/
    checkpoint_final.pt
    history.json
  prediction/
    prospectivity.png
    uncertainty.png
  manifest.json
```

主要看：

- `prediction/prospectivity.png`：成矿远景概率图。
- `prediction/uncertainty.png`：预测不确定性图。
- `sample_gan/generated_patches_preview.png`：GAN 生成 patch 预览，检查生成样本是否合理。

## 6. 注意事项

所有证据图层必须同尺寸、同坐标网格、同分辨率。

矿点坐标必须是像素坐标，不是经纬度或投影坐标。如果你现在只有经纬度/投影坐标，需要先转换到栅格 row/col。

GAN 生成样本是辅助增强，不应该完全替代真实矿点。建议做两组对比：

```text
1. 只用真实样本训练
2. 使用 GAN 增强样本训练
```

如果第二组在已知矿点回代、空间合理性、不确定性上更好，再采用增强结果。
