"""工具函数"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Logger:
    """简单日志记录器"""
    
    def __init__(self, log_file='training.log'):
        self.log_file = log_file
    
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')


def create_directories(dirs):
    """创建必要的目录"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 目录已创建: {dir_path}")


def save_metrics(metrics_dict, save_path):
    """保存评估指标为JSON"""
    # 将numpy值转换为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_dict = convert_to_serializable(metrics_dict)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 指标已保存: {save_path}")


def load_metrics(load_path):
    """从JSON加载评估指标"""
    with open(load_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics


def print_summary(summary_dict):
    """打印结果摘要"""
    print("\n" + "="*60)
    print("训练完成！结果摘要")
    print("="*60)
    
    for key, value in summary_dict.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*60)


def format_number(num, decimals=4):
    """格式化数字"""
    if isinstance(num, float):
        return f"{num:.{decimals}f}"
    return str(num)


def get_class_weights(y, method='balanced'):
    """计算类权重"""
    unique, counts = np.unique(y, return_counts=True)
    
    if method == 'balanced':
        total = len(y)
        weights = {cls: total / (len(unique) * count) 
                  for cls, count in zip(unique, counts)}
    elif method == 'inverse':
        weights = {cls: 1 / count for cls, count in zip(unique, counts)}
    else:
        weights = {cls: 1.0 for cls in unique}
    
    return weights


def print_data_info(X, y, set_name='Dataset'):
    """打印数据集信息"""
    print(f"\n{set_name} 信息:")
    print(f"  样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  类别分布: {np.bincount(y)}")
    print(f"  正样本比例: {np.mean(y):.2%}")
