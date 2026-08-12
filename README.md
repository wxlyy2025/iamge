# 矿点检测模型系统

## 项目简介

这是一个基于机器学习和深度学习的**矿点二分类检测系统**，用于区分矿点数据（标签=1）和非矿点数据（标签=0）。

**核心挑战：**
- 类严重不平衡：矿点样本 4,060个 vs 无矿点样本 ~40,000个
- 特征维度：31维
- 需要高召回率和精准率的平衡

## 📊 项目特点

✅ **完整的机器学习流程**
- 数据加载与预处理
- 类不平衡处理（SMOTE + 欠采样）
- 多种模型（传统ML + 深度学习）
- 科学的评估指标

✅ **多模型支持**
- 传统模型：逻辑回归、随机森林、梯度提升、XGBoost、LightGBM、SVM
- 深度学习：MLP、LSTM

✅ **生产级代码结构**
- 模块化设计
- 配置驱动
- 完整的日志记录
- 可视化评估报告

## 🏗️ 项目结构

```
mining-point-detection/
├── data/
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后数据
├── notebooks/
│   └── eda.ipynb              # 探索性数据分析
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
│   ├── models/                # 训练的模型文件
│   └── metrics/               # 评估指标
├── requirements.txt
├── README.md
└── main.py                    # 主程序入口
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将您的CSV数据放在 `data/raw/mining_data.csv`，格式如下：

```csv
feature_1,feature_2,...,feature_31,label
0.5,0.3,...,0.8,0
1.2,0.9,...,0.6,1
...
```

### 3. 配置调整

编辑 `config/config.yaml` 调整参数：

```yaml
data_path: './data/raw/mining_data.csv'
epochs: 100
batch_size: 32
learning_rate: 0.001
imbalance_method: 'smote_under'  # 类不平衡处理方法
```

### 4. 运行完整流程

```bash
python main.py
```

### 5. 查看结果

```
results/
├── plots/           # ROC曲线、PR曲线、混淆矩阵
├── models/          # 训练的模型文件
├── metrics/         # 详细评估指标
└── summary.json     # 结果摘要
```

## 📈 模型性能

| 模型 | 验证AUC | 测试AUC | F1-Score | 推荐指数 |
|------|--------|--------|----------|----------|
| XGBoost | - | - | - | ⭐⭐⭐⭐⭐ |
| LightGBM | - | - | - | ⭐⭐⭐⭐⭐ |
| 随机森林 | - | - | - | ⭐⭐⭐⭐ |
| MLP | - | - | - | ⭐⭐⭐⭐ |
| LSTM | - | - | - | ⭐⭐⭐ |

## 🎯 核心策略

### 1. 类不平衡处理

**方案：SMOTE + 欠采样（组合策略）**

```python
# 先过采样少数类（SMOTE）
# 再欠采样多数类（保持1:0.7比例）
from imblearn.pipeline import Pipeline
pipeline = Pipeline([
    ('smote', SMOTE(k_neighbors=5)),
    ('under', RandomUnderSampler(sampling_strategy=0.7))
])
```

**优势：**
- ✓ 防止过度采样导致过拟合
- ✓ 保留足够的多数类样本
- ✓ 提高模型泛化能力

### 2. 加权损失函数

深度学习中使用加权BCE损失处理不平衡：

```python
criterion = nn.BCELoss(weight=torch.tensor([pos_weight]))
```

### 3. 多模型融合

- 快速验证：使用XGBoost（训练快、效果好）
- 深度学习：MLP/LSTM（捕捉复杂特征关系）
- 集成方法：投票或加权平均

### 4. 评估指标

**选择原因：**
- **ROC-AUC**：不受类不平衡影响
- **PR-AUC**：重点关注少数类
- **F1-Score**：精准率和召回率平衡
- **Confusion Matrix**：详细的分类情况

### 5. 最优阈值

**不要使用默认0.5！**

```python
# 使用F1-Score或特异性寻找最优阈值
best_threshold = find_best_threshold(y_test, y_pred_proba, metric='f1')
y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
```

## 🔧 高级用法

### 单独训练特定模型

```python
from src.models import TraditionalModels
from src.train import TraditionalModelTrainer

models = TraditionalModels.get_models()
results = TraditionalModelTrainer.train_and_evaluate(
    models, X_train, y_train, X_val, y_val, X_test, y_test
)
```

### 自定义模型

```python
from src.models import MiningNN
import torch.nn as nn

class CustomModel(MiningNN):
    def __init__(self, input_dim=31):
        super().__init__(input_dim)
        # 自定义架构
        self.custom_layer = nn.Linear(32, 16)
```

### 推理和预测

```python
from src.train import DeepLearningTrainer

trainer = DeepLearningTrainer(model)
y_pred_proba = trainer.predict(X_new)
y_pred = (y_pred_proba >= 0.5).astype(int)
```

## 📊 数据分析

运行Jupyter Notebook进行EDA：

```bash
jupyter lab notebooks/eda.ipynb
```

分析内容：
- 特征分布
- 类别分布
- 缺失值
- 特征相关性
- 异常值检测

## 🐛 常见问题

### Q1: 模型总是预测为0（无矿点）

**原因：** 类不平衡导致模型偏向多数类

**解决方案：**
1. 调整 `pos_weight` 参数（增大权重）
2. 更激进的SMOTE策略
3. 调整预测阈值

### Q2: 如何改进模型性能？

1. **数据角度**
   - 收集更多矿点样本
   - 特征工程（创建新特征）
   - 数据清洗

2. **模型角度**
   - 调整超参数
   - 尝试不同的模型架构
   - 使用集成方法

3. **验证角度**
   - 使用K折交叉验证
   - 使用分层划分
   - 监控多个指标

### Q3: 部署到生产环境

```python
import pickle

# 保存模型
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 加载模型
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## 📚 参考资源

- [Imbalanced Learn Documentation](https://imbalanced-learn.org/)
- [PyTorch Official Guide](https://pytorch.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请在GitHub Issues中反馈。

---

**最后更新：** 2026年5月
**版本：** 1.0.0