# Mineral Prospect Model 

这是一个用于矿产远景预测/反演的多模型版本。输入为多源找矿证据图层，输出为每个像元的成矿远景概率图。

## 可选模型

训练时通过 `--model` 选择网络：

| 参数 | 说明 |
|---|---|
| `unet` | 基础 U-Net，适合小数据和快速基线。 |
| `resunet` | 轻量残差 U-Net，适合多源证据层。 |
| `resnet50_unet` | 本地实现的 ResNet-50 Bottleneck 编码器 + U-Net 解码器。 |
| `attention_unet` | 带注意力门的 U-Net，更关注矿点相关空间区域。 |
| `aspp_resunet` | ResUNet + DeepLab 风格 ASPP，多尺度上下文更强。 |
| `simple_cnn` | 轻量 CNN 基线，速度快，适合对照实验。 |

训练模式：

```powershell
--training-mode gan          # 生成器/反演网络 + PatchGAN 判别器
--training-mode supervised   # 只使用监督损失训练反演网络
```

## 环境

你当前指定的 Python：

```powershell
D:\anaconda\envs\deep\python.exe
```

## 一键测试

```powershell
cd D:\data\PyTorch-GAN-master\MineralProspectModelZoo
D:\anaconda\envs\deep\python.exe run_demo.py --python D:\anaconda\envs\deep\python.exe --model resnet50_unet --training-mode supervised
```

`resnet50_unet` 参数量较大，CPU 上建议先用 `supervised` 模式和较小 `base-channels` 跑通；正式训练再加大轮次。

## 真实数据训练

证据图层：

```text
evidences.npy  shape = [C, H, W]
```

矿点 CSV：

```csv
row,col,deposit
42,38,1
88,90,1
```

训练示例：

```powershell
D:\anaconda\envs\deep\python.exe train.py ^
  --evidence D:\your_data\evidences.npy ^
  --deposits D:\your_data\deposits.csv ^
  --out-dir runs\yulong_resnet50 ^
  --model resnet50_unet ^
  --training-mode gan ^
  --epochs 80 ^
  --patch-size 64 ^
  --samples-per-epoch 2048 ^
  --batch-size 4 ^
  --base-channels 16
```

预测：

```powershell
D:\anaconda\envs\deep\python.exe predict.py ^
  --evidence D:\your_data\evidences.npy ^
  --checkpoint runs\yulong_resnet50\checkpoint_final.pt ^
  --out-dir runs\yulong_resnet50_prediction ^
  --mc-samples 16
```

## 输出

- `checkpoint_final.pt`：模型权重
- `history.json`：训练损失
- `prospectivity.npy/png`：矿产远景概率图
- `uncertainty.npy/png`：多次随机前向预测的不确定性图
