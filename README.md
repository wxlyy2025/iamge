# 矿点检测模型系统

## 项目简介

这是一个完整的矿点检测机器学习系统，用于处理矿点数据的二分类问题。项目包含数据预处理、模型训练、评估和可视化等完整流程。

**关键特点：**
- ✅ 处理严重类不平衡（正样本4060，负样本4万）
- ✅ 多种模型支持（传统ML + 深度学习）
- ✅ 完整的训练-验证-测试框架
- ✅ 科学的评估指标
- ✅ 生产级代码结构

---

## 项目统计

- **总样本数**：约44,060个
- **矿点样本**（label=1）：4,060个
- **无矿点样本**（label=0）：约40,000个
- **特征维度**：31维
- **类不平衡比例**：1:10

---

## 项目结构

```
mining-point-detection/
├── data/
│   ├── raw/                    # 原始数据
│   │   └── mining_data.csv    # 输入数据文件
│   └── processed/              # 处理后数据
├── notebooks/
│   ├── eda.ipynb              # 探索性数据分析
│   └── demo.ipynb             # 完整演示
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载和预处理
│   ├── models.py              # 模型定义
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── utils.py               # 工具函数
├── config/
│   └── config.yaml            # 配置文件
├── results/
│   ├── plots/                 # 评估图表
│   ├── models/                # 训练的模型
│   ├── metrics/               # 详细指标
│   └── summary.json           # 结果摘要
├── requirements.txt           # 依赖包
├── main.py                    # 主程序入口
├── README.md                  # 项目说明
└── .gitignore                 # Git忽略文件
```

---

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/wxlyy2025/iamge.git
cd iamge

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

将您的CSV数据放在 `data/raw/mining_data.csv`，格式应如下：

```csv
feature_1,feature_2,...,feature_31,label
0.5,0.3,...,0.8,0
1.2,0.9,...,0.6,1
...
```

### 3. 运行完整流程

```bash
# 训练所有模型
python main.py

# 或在Jupyter中运行
jupyter notebook notebooks/demo.ipynb
```

### 4. 查看结果

所有结果保存在 `results/` 目录：
- `plots/` - ROC曲线、PR曲线、混淆矩阵等图表
- `models/` - 训练好的模型文件
- `metrics/` - 详细评估指标
- `summary.json` - 结果摘要

---

## 模型说明

### 传统机器学习模型

| 模型 | 优势 | 适用场景 |
|------|------|--------|
| **逻辑回归(LR)** | 简单、快速、可解释 | 基准模型 |
| **随机森林(RF)** | 鲁棒、处理非线性 | 中等规模数据 |
| **梯度提升(GB)** | 高精度、特征重要性 | 复杂数据 |
| **XGBoost** | 超高精度、处理不平衡 | 推荐首选 |
| **LightGBM** | 快速、内存高效 | 大规模数据 |
| **SVM** | 高维数据表现好 | 小规模数据 |

### 深度学习模型

| 模型 | 架构 | 适用场景 |
|------|------|--------|
| **MLP** | 4层全连接 + BatchNorm | 标准特征输入 |
| **LSTM** | 2层LSTM + 全连接 | 序列特征 |

---

## 类不平衡处理方案

本项目使用多种策略处理严重的类不平衡问题：

### 1. **数据级方案**
- **SMOTE**：为少数类(矿点)生成合成样本
- **欠采样**：从多数类中随机删除样本
- **结合策略**：先SMOTE后欠采样（推荐）

### 2. **算法级方案**
- **加权损失函数**：给少数类更高的权重
- **阈值调整**：使用F1或Specificity找最优阈值
- **自适应学习率**：动态调整学习率

### 3. **评估级方案**
- **使用ROC-AUC**：不受类不平衡影响
- **使用PR-AUC**：对少数类更敏感
- **使用F1-Score**：平衡精准率和召回率
- **混淆矩阵**：详细了解误分类情况

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `imbalance_method` | `smote_under` | 类不平衡处理方法 |
| `batch_size` | 32 | 批处理大小 |
| `learning_rate` | 0.001 | 学习率 |
| `epochs` | 100 | 训练轮数 |
| `dropout_rate` | 0.3 | 丢弃率（防过拟合） |
| `patience` | 15 | 早停耐心度 |
| `pos_weight` | 10 | 正样本权重 |

在 `config/config.yaml` 中修改这些参数。

---

## 评估指标解释

- **Accuracy** - 总体正确率（受类不平衡影响）
- **Precision** - 预测为矿点中真正是矿点的比例
- **Recall** - 实际矿点中被正确预测的比例
- **F1-Score** - Precision和Recall的调和平均数
- **ROC-AUC** - 不受类不平衡影响，综合评估能力
- **PR-AUC** - 对少数类更敏感

### 选择指标的建议：

- 💡 **关心漏检**（不能漏掉矿点）→ 使用**Recall**
- 💡 **关心误报**（不能误报）→ 使用**Precision**
- 💡 **综合考虑** → 使用**F1-Score**
- 💡 **整体评估** → 使用**ROC-AUC**

---

## 使用示例

### 简单预测

```python
from src.data_loader import MiningDataLoader
from src.models import TraditionalModels
import numpy as np

# 加载数据
loader = MiningDataLoader()
X, y = loader.preprocess(df, feature_cols)

# 训练XGBoost
models = TraditionalModels.get_models()
xgb_model = models['XGB']
xgb_model.fit(X_train, y_train)

# 预测
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"预测准确率: {(y_pred == y_test).mean():.4f}")
```

### 找最优阈值

```python
from src.evaluate import ModelEvaluator

best_threshold = ModelEvaluator.find_best_threshold(
    y_test, y_pred_proba, metric='f1'
)

y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
print(f"最优阈值: {best_threshold:.3f}")
```

---

## 常见问题

### Q1: 为什么默认阈值不是0.5？
**A**: 类不平衡情况下，0.5通常不是最优的。建议用F1-Score或Specificity找最优阈值。

### Q2: 如何选择合适的模型？
**A**: 
- 首先用XGBoost/LightGBM（快速、高精度）
- 如果需要解释性，用随机森林
- 如果有GPU和大量数据，考虑深度学习

### Q3: 我的数据不足44k怎么办？
**A**: 
- 使用SMOTE进行过采样
- 考虑数据增强或获取更多数据
- 或使用更简单的模型

### Q4: 如何处理新的矿点位置？
**A**: 
- 保存训练好的模型
- 使用相同的特征工程
- 调用 `predict()` 方法

---

## 性能基准

基于4060个正样本和4万个负样本的预期性能：

| 模型 | ROC-AUC | F1-Score | Recall | 训练时间 |
|------|---------|----------|--------|--------|
| XGBoost | ~0.92 | ~0.75 | ~0.80 | ~5min |
| LightGBM | ~0.91 | ~0.73 | ~0.78 | ~2min |
| 随机森林 | ~0.89 | ~0.70 | ~0.75 | ~3min |
| MLP | ~0.88 | ~0.68 | ~0.72 | ~10min |

*注：实际性能取决于数据质量和特征*

---

## 最佳实践

✅ **Do's:**
- ✓ 使用ROC-AUC而不是Accuracy评估
- ✓ 调整阈值而不是相信默认0.5
- ✓ 使用交叉验证评估模型
- ✓ 定期更新模型
- ✓ 监控特征重要性

❌ **Don'ts:**
- ✗ 不要用Accuracy评估不平衡数据
- ✗ 不要忽视类不平衡问题
- ✗ 不要过度拟合（使用正则化）
- ✗ 不要忽视数据质量

---

## 贡献指南

欢迎提交改进建议和报告问题！

---

## 许可证

MIT License

---

## 联系方式

- **开发者**: wxlyy2025
- **邮箱**: [您的邮箱]
- **GitHub**: https://github.com/wxlyy2025

---

**最后更新**: 2026年5月
