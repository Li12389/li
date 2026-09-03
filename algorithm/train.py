"""
模型训练脚本
功能：加载预处理后的训练集，训练随机森林分类模型，评估模型性能，保存模型和特征重要性
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 确保目录存在
os.makedirs('model', exist_ok=True)

print("=" * 60)
print("开始训练随机森林分类模型...")
print("=" * 60)

# ========== 1. 加载预处理后的数据 ==========
print("\n[1] 加载训练集和测试集...")

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').iloc[:, 0]
y_test = pd.read_csv('data/y_test.csv').iloc[:, 0]

feature_cols = X_train.columns.tolist()
print(f"   训练集: {X_train.shape[0]} 条 × {X_train.shape[1]} 特征")
print(f"   测试集: {X_test.shape[0]} 条 × {X_test.shape[1]} 特征")
print(f"   特征列: {feature_cols}")
print(f"   训练集标签分布:\n{y_train.value_counts()}")
print(f"   测试集标签分布:\n{y_test.value_counts()}")

# ========== 2. 特征名映射（简洁API名 -> 原始列名） ==========
print("\n[2] 建立特征名映射...")

FEATURE_MAP = {
    'arc_voltage': 'Arc Voltage (V)',
    'welding_current': 'Weld Current (A)',
    'welding_speed': 'Weld Speed (cm/min)',
    'wire_feed_speed': 'Wire Feed Speed (m/min)',
    'gas_flow_rate': 'Gas Flow Rate (L/min)',
    'torch_angle': 'Torch Angle (°)',
    'base_metal_temperature': 'Base Metal Temp (°C)'
}

# 验证映射完整性
for api_name, orig_name in FEATURE_MAP.items():
    assert orig_name in feature_cols, f"特征 {orig_name} 不在数据列中"
print("   ✓ 特征映射验证通过")

# 保存特征映射，供后端预测使用
with open('model/feature_map.json', 'w', encoding='utf-8') as f:
    json.dump(FEATURE_MAP, f, ensure_ascii=False, indent=2)
print("   ✓ 特征映射已保存: model/feature_map.json")

# ========== 3. 训练随机森林模型 ==========
print("\n[3] 训练随机森林分类模型...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("   ✓ 模型训练完成")

# ========== 4. 模型评估 ==========
print("\n[4] 模型评估...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"   准确率 (Accuracy):  {accuracy:.4f}")
print(f"   精确率 (Precision): {precision:.4f}")
print(f"   召回率 (Recall):    {recall:.4f}")
print(f"   F1-Score:           {f1:.4f}")
print(f"\n   混淆矩阵:")
print(f"              预测无缺陷  预测有缺陷")
print(f"   实际无缺陷    {cm[0][0]:>6}      {cm[0][1]:>6}")
print(f"   实际有缺陷    {cm[1][0]:>6}      {cm[1][1]:>6}")
print(f"\n   分类报告:")
print(classification_report(y_test, y_pred, target_names=['无缺陷', '有缺陷'], zero_division=0))

# 保存评估指标
metrics = {
    'accuracy': round(accuracy, 4),
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4),
    'confusion_matrix': cm.tolist(),
    'train_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'feature_count': int(X_train.shape[1])
}
with open('model/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("   ✓ 评估指标已保存: model/metrics.json")

# ========== 5. 特征重要性分析 ==========
print("\n[5] 特征重要性分析...")

importances = rf_model.feature_importances_
feature_importance = []
for i, col in enumerate(feature_cols):
    # 找到对应的API名称
    api_name = next((k for k, v in FEATURE_MAP.items() if v == col), col)
    feature_importance.append({
        'feature': api_name,
        'feature_name_cn': {
            'arc_voltage': '电弧电压',
            'welding_current': '焊接电流',
            'welding_speed': '焊接速度',
            'wire_feed_speed': '送丝速度',
            'gas_flow_rate': '气体流量',
            'torch_angle': '焊枪角度',
            'base_metal_temperature': '母材温度'
        }.get(api_name, col),
        'importance': round(float(importances[i]), 4)
    })

# 按重要性降序排序
feature_importance.sort(key=lambda x: x['importance'], reverse=True)

print("   特征重要性排序（降序）:")
for i, item in enumerate(feature_importance, 1):
    print(f"   {i}. {item['feature_name_cn']} ({item['feature']}): {item['importance']:.4f}")

# 保存特征重要性
with open('model/feature_importance.json', 'w', encoding='utf-8') as f:
    json.dump(feature_importance, f, ensure_ascii=False, indent=2)
print("   ✓ 特征重要性已保存: model/feature_importance.json")

# ========== 6. 保存模型 ==========
print("\n[6] 保存模型...")

joblib.dump(rf_model, 'model/classifier.pkl')
print("   ✓ 分类模型已保存: model/classifier.pkl")

print("\n" + "=" * 60)
print("模型训练完成！")
print(f"   模型文件: model/classifier.pkl")
print(f"   标准化器: model/scaler.pkl")
print(f"   特征映射: model/feature_map.json")
print(f"   特征重要性: model/feature_importance.json")
print(f"   评估指标: model/metrics.json")
print("=" * 60)
