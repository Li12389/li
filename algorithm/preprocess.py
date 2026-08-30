"""
数据预处理脚本
功能：加载焊接缺陷数据集，进行标准化处理，划分训练集和测试集
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# 创建必要的目录
os.makedirs('model', exist_ok=True)

print("=" * 50)
print("开始数据预处理...")
print("=" * 50)

# ========== 1. 加载原始数据 ==========
print("\n[1] 加载原始数据...")

csv_path = 'data/1_Robotic_welding_edge_dataset_v2.csv'
df = pd.read_csv(csv_path)
print(f"   ✓ 已加载数据文件: {csv_path}")
print(f"   数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"   列名: {df.columns.tolist()}")

print(f"\n   前5行数据预览:")
print(df.head())

# ========== 2. 数据探索 ==========
print("\n[2] 数据探索...")

print(f"   缺失值统计:\n{df.isnull().sum()}")

print(f"\n   数据类型:\n{df.dtypes}")

label_col = df.columns[-1]
print(f"\n   标签列: {label_col}")
print(f"   标签分布:\n{df[label_col].value_counts()}")

# ========== 3. 分离特征和标签 ==========
print("\n[3] 分离特征和标签...")

feature_cols = [col for col in df.columns if col != label_col]
X = df[feature_cols]
y = df[label_col]

print(f"   特征数: {X.shape[1]}")
print(f"   特征列: {feature_cols}")

# ========== 4. 标准化处理 ==========
print("\n[4] 标准化处理 (StandardScaler)...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   已完成标准化，特征矩阵形状: {X_scaled.shape}")

joblib.dump(scaler, 'model/scaler.pkl')
print("   ✓ 标准化器已保存: model/scaler.pkl")

# ========== 5. 按 7:3 划分训练集和测试集 ==========
print("\n[5] 划分训练集和测试集 (7:3)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

print(f"   训练集: {X_train.shape[0]} 条")
print(f"   测试集: {X_test.shape[0]} 条")

# ========== 6. 保存处理后的数据 ==========
print("\n[6] 保存处理后的数据...")

df_processed = pd.DataFrame(X_scaled, columns=feature_cols)
df_processed[label_col] = y.values
df_processed.to_csv('data/welding_data_processed.csv', index=False)
print("   ✓ 标准化后完整数据: data/welding_data_processed.csv")

pd.DataFrame(X_train, columns=feature_cols).to_csv('data/X_train.csv', index=False)
pd.DataFrame(X_test, columns=feature_cols).to_csv('data/X_test.csv', index=False)
pd.DataFrame(y_train, columns=[label_col]).to_csv('data/y_train.csv', index=False)
pd.DataFrame(y_test, columns=[label_col]).to_csv('data/y_test.csv', index=False)
print("   ✓ 训练集: data/X_train.csv, data/y_train.csv")
print("   ✓ 测试集: data/X_test.csv, data/y_test.csv")

print("\n" + "=" * 50)
print("数据预处理完成！")
print("=" * 50)