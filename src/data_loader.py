import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class MiningDataLoader:
    """矿点数据加载和预处理类"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def load_data(self, filepath):
        """加载数据"""
        df = pd.read_csv(filepath)
        print(f"原始数据形状: {df.shape}")
        print(f"类别分布:\n{df['label'].value_counts()}")
        missing_count = df.isnull().sum().sum()
        print(f"缺失值总数: {missing_count}")
        return df
    
    def preprocess(self, df, feature_cols, label_col='label'):
        """数据预处理"""
        # 处理缺失值
        df_clean = df.dropna()
        print(f"删除缺失值后形状: {df_clean.shape}")
        
        # 特征和标签分离
        X = df_clean[feature_cols].values
        y = df_clean[label_col].values
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"特征标准化完成，形状: {X_scaled.shape}")
        return X_scaled, y
    
    def handle_imbalance(self, X_train, y_train, method='smote_under'):
        """处理类不平衡问题
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            method: 处理方法
                - 'smote': 过采样少数类
                - 'under': 欠采样多数类
                - 'smote_under': 组合策略（推荐）
                - 'none': 不处理
        
        Returns:
            X_resampled, y_resampled: 处理后的数据
        """
        if method == 'none':
            return X_train, y_train
        
        print(f"\n原始训练集类别分布: {np.bincount(y_train)}")
        
        if method == 'smote':
            # 仅使用SMOTE过采样
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
        elif method == 'under':
            # 欠采样
            under = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = under.fit_resample(X_train, y_train)
            
        elif method == 'smote_under':
            # 组合策略：先过采样后欠采样（推荐）
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=self.random_state, k_neighbors=5)),
                ('under', RandomUnderSampler(random_state=self.random_state, 
                                            sampling_strategy=0.7))
            ])
            X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        
        print(f"处理后训练集类别分布: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """分割数据集为训练、验证和测试集"""
        # 先分出测试集（保持原始比例）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # 再分出验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"\n数据集划分:")
        print(f"训练集: {X_train.shape}, 类别: {np.bincount(y_train)}")
        print(f"验证集: {X_val.shape}, 类别: {np.bincount(y_val)}")
        print(f"测试集: {X_test.shape}, 类别: {np.bincount(y_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
